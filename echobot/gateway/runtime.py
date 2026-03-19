from __future__ import annotations

import asyncio
import logging

from ..channels import InboundMessage, MessageBus, OutboundMessage
from ..channels.types import DeliveryTarget
from ..commands.bindings import GatewayCommandContext, dispatch_gateway_command
from ..runtime.scheduled_tasks import (
    build_cron_job_executor as build_shared_cron_job_executor,
    build_heartbeat_executor as build_shared_heartbeat_executor,
)
from ..runtime.bootstrap import RuntimeContext
from ..runtime.session_service import SessionLifecycleService
from .delivery import DeliveryStore
from .route_sessions import RouteSessionStore
from .session_service import GatewaySessionService


logger = logging.getLogger(__name__)


class GatewayRuntime:
    def __init__(
        self,
        context: RuntimeContext,
        bus: MessageBus,
        session_service: GatewaySessionService | None = None,
        delivery_store: DeliveryStore | None = None,
        route_session_store: RouteSessionStore | None = None,
        *,
        max_inflight_messages: int = 32,
    ) -> None:
        self._context = context
        self._bus = bus
        if session_service is None:
            delivery_store = delivery_store or DeliveryStore(
                context.workspace / ".echobot" / "delivery.json",
            )
            route_session_store = route_session_store or RouteSessionStore(
                context.workspace / ".echobot" / "route_sessions.json",
            )
            core_session_service = SessionLifecycleService(
                context.session_store,
                context.agent_session_store,
                coordinator=context.coordinator,
            )
            session_service = GatewaySessionService(
                core_session_service,
                route_session_store=route_session_store,
                delivery_store=delivery_store,
            )
        self._session_service = session_service
        self._inflight_tasks: set[asyncio.Task[None]] = set()
        self._inflight_semaphore = asyncio.Semaphore(max(max_inflight_messages, 1))
        self._route_locks: dict[str, asyncio.Lock] = {}
        self._route_locks_guard = asyncio.Lock()

    async def run(self) -> None:
        self._context.cron_service.on_job = self._build_cron_job_executor()
        if self._context.heartbeat_service is not None:
            self._context.heartbeat_service.on_execute = (
                self._build_heartbeat_executor()
            )
            self._context.heartbeat_service.on_notify = self._notify_latest

        await self._context.cron_service.start()
        if self._context.heartbeat_service is not None:
            await self._context.heartbeat_service.start()

        logger.info("Gateway runtime started")
        try:
            while True:
                await self._inflight_semaphore.acquire()
                message = await self._bus.consume_inbound()
                task = asyncio.create_task(self._handle_inbound_message_task(message))
                self._inflight_tasks.add(task)
                task.add_done_callback(self._inflight_tasks.discard)
        finally:
            await self._shutdown()

    async def handle_inbound_message(self, message: InboundMessage) -> None:
        route_lock = await self._route_lock(message.route_key)
        async with route_lock:
            await self._handle_inbound_message(message)

    async def _handle_inbound_message_task(self, message: InboundMessage) -> None:
        try:
            await self.handle_inbound_message(message)
        finally:
            self._inflight_semaphore.release()

    async def _handle_inbound_message(self, message: InboundMessage) -> None:
        route_key = message.route_key
        command_result = await dispatch_gateway_command(
            GatewayCommandContext(
                coordinator=self._context.coordinator,
                workspace=self._context.workspace,
                session_service=self._session_service,
                route_key=route_key,
                address=message.address,
                metadata=message.metadata,
            ),
            message.text,
        )
        if command_result is not None:
            await self._bus.publish_outbound(
                OutboundMessage(
                    address=message.address,
                    text=command_result.text,
                    metadata=dict(message.metadata),
                )
            )
            return

        route_session = await self._session_service.current_route_session(
            route_key,
        )
        await self._session_service.remember_delivery_target(
            route_session.session_name,
            message.address,
            message.metadata,
        )
        try:
            execution = await self._context.coordinator.handle_user_turn(
                route_session.session_name,
                message.text,
                image_urls=message.image_urls,
                completion_callback=self._completion_callback_for_session(
                    route_session.session_name,
                ),
            )
            content = execution.response_text.strip()
            await self._session_service.touch_route_session(
                route_key,
                route_session.session_name,
                updated_at=execution.session.updated_at,
            )
            if not content and execution.delegated and not execution.completed:
                return
            if not content:
                content = "Model returned no text content."
        except ValueError as exc:
            content = str(exc)
        except RuntimeError as exc:
            content = f"Request failed: {exc}"
        await self._bus.publish_outbound(
            OutboundMessage(
                address=message.address,
                text=content,
                metadata=dict(message.metadata),
            )
        )

    def _completion_callback_for_session(
        self,
        session_name: str,
    ):
        async def notify(job) -> None:
            await self._publish_session_response(
                session_name,
                job.final_response,
                metadata={
                    "async_result": True,
                    "job_id": job.job_id,
                    "job_status": job.status,
                },
            )

        return notify

    def _build_cron_job_executor(self):
        return build_shared_cron_job_executor(
            self._context.session_runner,
            self._context.coordinator,
            self._notify_schedule,
        )

    def _build_heartbeat_executor(self):
        return build_shared_heartbeat_executor(self._context.session_runner)

    async def _notify_session(
        self,
        session_name: str,
        content: str,
        *,
        kind: str,
        title: str,
    ) -> None:
        target = await self._session_service.get_session_target(session_name)
        await self._publish_notification(
            target,
            content,
            kind=kind,
            title=title,
        )

    async def _publish_session_response(
        self,
        session_name: str,
        content: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        target = await self._session_service.get_session_target(session_name)
        if target is None:
            logger.info("[reply] %s", content)
            return
        next_metadata = dict(target.metadata)
        if metadata is not None:
            next_metadata.update(metadata)
        await self._bus.publish_outbound(
            OutboundMessage(
                address=target.address,
                text=content,
                metadata=next_metadata,
            )
        )

    async def _notify_latest(self, content: str) -> None:
        target = await self._session_service.get_latest_target()
        await self._publish_notification(
            target,
            content,
            kind="heartbeat",
            title="Periodic check-in",
        )

    async def _publish_notification(
        self,
        target: DeliveryTarget | None,
        content: str,
        *,
        kind: str,
        title: str,
    ) -> None:
        if target is None:
            logger.info("[%s] %s", kind, title)
            for line in content.splitlines() or [content]:
                logger.info("[%s] %s", kind, line)
            return
        metadata = dict(target.metadata)
        metadata["scheduled"] = True
        metadata["schedule_kind"] = kind
        metadata["schedule_title"] = title
        await self._bus.publish_outbound(
            OutboundMessage(
                address=target.address,
                text=content,
                metadata=metadata,
            )
        )

    async def _notify_schedule(
        self,
        session_name: str,
        kind: str,
        title: str,
        content: str,
    ) -> None:
        await self._notify_session(
            session_name,
            content,
            kind=kind,
            title=title,
        )

    async def _shutdown(self) -> None:
        tasks = list(self._inflight_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._context.cron_service.stop()
        if self._context.heartbeat_service is not None:
            await self._context.heartbeat_service.stop()
        await self._context.coordinator.close()
        if self._context.memory_support is not None:
            await self._context.memory_support.close()

    async def _route_lock(self, route_key: str) -> asyncio.Lock:
        async with self._route_locks_guard:
            lock = self._route_locks.get(route_key)
            if lock is None:
                lock = asyncio.Lock()
                self._route_locks[route_key] = lock
            return lock
