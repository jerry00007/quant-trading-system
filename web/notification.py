"""通知服务 — 支持邮件和浏览器推送

邮件: SMTP发送每日操作信号
浏览器: Web Push API (需前端配合)

配置存储在 data/sqlite/portfolio.db 的 settings 表中。
"""
import json
import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def _get_settings_db():
    import sqlite3
    db_path = PROJECT_ROOT / "data" / "sqlite" / "portfolio.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def get_notification_config() -> dict:
    conn = _get_settings_db()
    row = conn.execute("SELECT value FROM settings WHERE key='notification_config'").fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return {
        "enabled": False,
        "email_enabled": False,
        "email_smtp": "",
        "email_port": 465,
        "email_user": "",
        "email_pass": "",
        "email_to": "",
        "browser_enabled": True,
        "notify_time": "09:25",
        "notify_on_signal": True,
        "notify_on_regime_change": True,
    }


def save_notification_config(config: dict):
    conn = _get_settings_db()
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        ("notification_config", json.dumps(config, ensure_ascii=False))
    )
    conn.commit()
    conn.close()


def format_signal_email(signals: list, portfolio: dict, regime: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_assets = portfolio.get("total_assets", 0)
    total_pnl = portfolio.get("total_pnl", 0)
    total_pnl_pct = portfolio.get("total_pnl_pct", 0)

    regime_text = f"{regime.get('regime', 'neutral')} ({regime.get('description', '')})" if regime else "未知"

    html = f"""
<html><body style="font-family: -apple-system, sans-serif; background: #0f1117; color: #e8eaed; padding: 20px;">
<div style="max-width: 600px; margin: 0 auto;">
  <h2 style="color: #00d4aa;">📊 每日操作信号</h2>
  <p style="color: #8b8fa3;">生成时间: {now} | 状态: {regime_text}</p>

  <div style="background: #1a1d29; border-radius: 8px; padding: 16px; margin: 16px 0;">
    <h3 style="color: #e8eaed; margin: 0 0 8px;">资产概况</h3>
    <p>总资产: ¥{total_assets:,.0f} | 收益: {total_pnl_pct:+.1f}% (¥{total_pnl:+,.0f})</p>
  </div>
"""
    if signals:
        html += '<div style="background: #1a1d29; border-radius: 8px; padding: 16px; margin: 16px 0;">'
        html += '<h3 style="color: #00d4aa; margin: 0 0 12px;">📋 今日操作</h3>'
        for s in signals:
            action_color = "#00d4aa" if s.get("action") == "buy" else "#ff4757"
            action_text = "买入" if s.get("action") == "buy" else "卖出"
            name = s.get("name", s.get("symbol", ""))
            price = s.get("reference_price", s.get("price", 0))
            reason = s.get("reason", "")
            html += f'''
            <div style="border-left: 3px solid {action_color}; padding: 8px 12px; margin: 8px 0; background: #141620;">
              <span style="color: {action_color}; font-weight: bold;">{action_text}</span>
              <span style="color: #e8eaed;">{name}</span>
              <span style="color: #8b8fa3;">@ ¥{price:.3f}</span>
              <div style="color: #5a5e72; font-size: 12px;">{reason}</div>
            </div>'''
        html += '</div>'
    else:
        html += '<div style="background: #1a1d29; border-radius: 8px; padding: 16px;"><p>今日无需操作，继续持仓。</p></div>'

    html += """
  <div style="color: #5a5e72; font-size: 12px; margin-top: 20px;">
    — 量化交易系统自动生成，请结合市场情况判断后操作
  </div>
</div></body></html>"""
    return html


def send_email_notification(subject: str, html_body: str, config: dict = None) -> bool:
    if config is None:
        config = get_notification_config()

    if not config.get("email_enabled"):
        return False

    smtp = config.get("email_smtp", "")
    port = config.get("email_port", 465)
    user = config.get("email_user", "")
    passwd = config.get("email_pass", "")
    to = config.get("email_to", "")

    if not all([smtp, user, passwd, to]):
        logger.warning("Email config incomplete")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP_SSL(smtp, port) as server:
            server.login(user, passwd)
            server.sendmail(user, to.split(","), msg.as_string())

        logger.info(f"Email sent to {to}")
        return True
    except Exception as e:
        logger.error(f"Email failed: {e}")
        return False


def get_scheduler_config() -> dict:
    conn = _get_settings_db()
    row = conn.execute("SELECT value FROM settings WHERE key='scheduler_config'").fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return {
        "data_update_time": "15:30",
        "signal_generate_time": "15:35",
        "notification_time": "09:25",
        "enabled": False,
    }


def save_scheduler_config(config: dict):
    conn = _get_settings_db()
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        ("scheduler_config", json.dumps(config, ensure_ascii=False))
    )
    conn.commit()
    conn.close()
