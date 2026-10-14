from __future__ import annotations

import asyncio
import logging
import os

import httpx
from dotenv import load_dotenv
from task_4.rag import RAGService
from task_4.rag.models import RAGAnswer
from telegram import Update
from telegram.constants import ChatAction
from telegram.error import NetworkError
from telegram.error import TimedOut
from telegram.ext import Application
from telegram.ext import CommandHandler
from telegram.ext import ContextTypes
from telegram.ext import MessageHandler
from telegram.ext import filters
from telegram.request import HTTPXRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def get_telegram_token() -> str:
    load_dotenv()

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN",
        "",
    ).strip()

    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required.")

    return token


def make_request(
    *,
    read_timeout: float,
) -> HTTPXRequest:
    """
    Force IPv6 only when explicitly requested.

    This keeps the project portable while still supporting environments where
    api.telegram.org is reachable only over IPv6.
    """

    force_ipv6 = (
        os.getenv(
            "TELEGRAM_FORCE_IPV6",
            "false",
        ).casefold()
        == "true"
    )

    kwargs = {}

    if force_ipv6:
        kwargs["transport"] = httpx.AsyncHTTPTransport(
            local_address="::",
            retries=1,
        )

    return HTTPXRequest(
        connect_timeout=10,
        read_timeout=read_timeout,
        write_timeout=30,
        pool_timeout=10,
        httpx_kwargs=kwargs,
    )


def render_answer(
    result: RAGAnswer,
) -> str:
    text = result.text

    if result.sources:
        lines = []

        for index, chunk in enumerate(
            result.sources,
            start=1,
        ):
            lines.append(
                f"[{index}] "
                f"{chunk.title} - "
                f"{chunk.source}, "
                f"chunk {chunk.chunk_index}, "
                f"similarity="
                f"{chunk.similarity:.3f}"
            )

        text += "\n\nSearch sources:\n" + "\n".join(lines)

    return text


async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if update.message:
        await update.message.reply_text("Hi! Ask a question about the witcher lore.")


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if not update.message or not update.message.text:
        return

    question = update.message.text.strip()

    if not question:
        return

    rag: RAGService = context.application.bot_data["rag"]

    # "Typing..." is a cosmetic request and must not break the actual RAG call.
    try:
        await update.message.chat.send_action(ChatAction.TYPING)
    except (TimedOut, NetworkError):
        logger.warning("Could not send Telegram typing action")

    try:
        result = await asyncio.to_thread(
            rag.ask,
            question,
        )

        await update.message.reply_text(render_answer(result))

    except Exception as exc:
        logger.exception("Failed to process message")

        await update.message.reply_text("Failed to handle request: " f"{type(exc).__name__}: {exc}")


async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    logger.exception(
        "Telegram handler error",
        exc_info=context.error,
    )


def main() -> None:
    token = get_telegram_token()
    rag = RAGService.from_env()

    application = (
        Application.builder()
        .token(token)
        .request(
            make_request(
                read_timeout=30,
            )
        )
        .get_updates_request(
            make_request(
                read_timeout=60,
            )
        )
        .build()
    )

    application.bot_data["rag"] = rag

    application.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    application.add_error_handler(error_handler)

    print("Telegram bot started.")

    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
