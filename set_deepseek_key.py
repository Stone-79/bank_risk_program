from __future__ import annotations

import getpass
import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "bank_risk_users.db"
SETTING_KEY = "deepseek_api_key"


def main() -> None:
    key = getpass.getpass("请输入 DeepSeek API Key（输入时不会显示）：").strip()
    if not key:
        raise SystemExit("未输入 API Key，配置已取消。")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_time TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO app_settings(key,value,updated_time) VALUES (?,?,?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_time=excluded.updated_time
            """,
            (SETTING_KEY, key, datetime.now().isoformat(timespec="seconds")),
        )
    print("DeepSeek API Key 已写入系统配置。")


if __name__ == "__main__":
    main()
