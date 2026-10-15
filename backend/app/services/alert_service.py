"""
Officer alerting service — significant-detection evaluation + Twilio SMS.

Design rules
------------
- Alerting is best-effort: any failure here (missing package, bad creds, network)
  is logged and swallowed. Inference and the API must never fail because of it.
- One alert per investigation. Re-running inference on an already-alerted
  investigation does nothing.
- Only fires when the configured detection condition is met — never per
  intermediate processing step.
- Twilio credentials come from the environment only; they are never returned to
  any API caller.
"""

from __future__ import annotations

import datetime as _dt
import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.alert import Alert, NotificationSettings
from app.services import severity as severity_mod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Twilio readiness (env only)
# ---------------------------------------------------------------------------


def _twilio_package_installed() -> bool:
    try:
        import twilio  # noqa: F401  (availability probe)

        return True
    except Exception:
        return False


def twilio_status(recipients: list[str], alerting_enabled: bool) -> dict:
    """Non-secret readiness summary for the officer UI."""
    return {
        "package_installed": _twilio_package_installed(),
        "account_configured": bool(
            (settings.TWILIO_ACCOUNT_SID or "").strip()
            and (settings.TWILIO_AUTH_TOKEN or "").strip()
        ),
        "from_number_set": bool((settings.TWILIO_FROM_NUMBER or "").strip()),
        "recipients_count": len(recipients),
        "alerting_enabled": alerting_enabled,
    }


def _send_sms(to: str, body: str) -> tuple[bool, str]:
    """Send one SMS. Returns ``(ok, detail)`` and never raises.

    Supports both auth styles:
    - Account SID (``AC…``) + Auth Token.
    - API Key SID (``SK…``) + API Key Secret, with the Account SID supplied
      separately in ``TWILIO_ACCOUNT_SID_AC`` (or ``TWILIO_ACCOUNT_SID`` if that
      itself is an ``AC…`` value).
    """
    sid = (settings.TWILIO_ACCOUNT_SID or "").strip()
    token = (settings.TWILIO_AUTH_TOKEN or "").strip()
    from_number = (settings.TWILIO_FROM_NUMBER or "").strip()
    account_sid_ac = (getattr(settings, "TWILIO_ACCOUNT_SID_AC", "") or "").strip()

    if not (sid and token and from_number):
        return False, "twilio_not_configured"

    try:
        from twilio.rest import Client  # noqa: PLC0415
    except Exception:
        return False, "twilio_package_not_installed"

    try:
        if sid.startswith("AC"):
            client = Client(sid, token)
        elif sid.startswith("SK"):
            acct = account_sid_ac or (sid if sid.startswith("AC") else "")
            if not acct.startswith("AC"):
                return False, (
                    "twilio_api_key_needs_account_sid: TWILIO_ACCOUNT_SID looks like an "
                    "API Key (SK…). Either set TWILIO_ACCOUNT_SID to your Account SID "
                    "(AC…) + Auth Token, or add TWILIO_ACCOUNT_SID_AC=AC… alongside the "
                    "API key + secret."
                )
            client = Client(sid, token, acct)
        else:
            client = Client(sid, token)
        msg = client.messages.create(to=to, from_=from_number, body=body)
        return True, f"sid={getattr(msg, 'sid', 'unknown')}"
    except Exception as exc:  # network / auth / invalid number
        logger.warning("Twilio send to %s failed: %s", to, exc)
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Notification settings (single row, id=1, seeded from env)
# ---------------------------------------------------------------------------


def _clean_numbers(raw: str) -> list[str]:
    """Split a comma list into plausible E.164 numbers.

    Tolerates a stray inline ``#`` comment or descriptive text in the .env value
    by keeping only tokens that look like phone numbers (optional ``+`` then
    digits/spaces/dashes, 7-15 digits).
    """
    out: list[str] = []
    for tok in (raw or "").split(","):
        tok = tok.split("#", 1)[0].strip()
        if not tok:
            continue
        digits = "".join(ch for ch in tok if ch.isdigit())
        if 7 <= len(digits) <= 15 and tok.lstrip("+").replace(" ", "").replace("-", "").isdigit():
            out.append("+" + digits if not tok.startswith("+") else "+" + digits)
    return out


def _env_recipients() -> list[str]:
    return _clean_numbers(settings.ALERT_RECIPIENTS)


async def get_or_create_settings(db: AsyncSession) -> NotificationSettings:
    row = await db.get(NotificationSettings, 1)
    if row is None:
        row = NotificationSettings(
            id=1,
            enabled=bool(settings.ALERT_ENABLED),
            min_confidence=float(settings.ALERT_MIN_CONFIDENCE),
            min_positive_tiles=int(settings.ALERT_MIN_POSITIVE_TILES),
            recipients=",".join(_env_recipients()),
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
    return row


def settings_recipients(row: NotificationSettings) -> list[str]:
    return _clean_numbers(row.recipients)


# ---------------------------------------------------------------------------
# Detection significance
# ---------------------------------------------------------------------------


@dataclass
class AlertDecision:
    significant: bool
    severity: str
    max_confidence: float
    positive_tiles: int
    area_km2: float
    reason: str


def evaluate_detection(
    ml_result: dict | None,
    min_confidence: float,
    min_positive_tiles: int,
) -> AlertDecision:
    """Decide whether an ``ml_result`` is significant enough to alert on."""
    s = severity_mod.summarise_detection(ml_result)
    significant = (
        s["has_detection"]
        and s["positive_tiles"] >= min_positive_tiles
        and s["max_confidence"] >= min_confidence
    )
    if significant:
        reason = (
            f"{s['positive_tiles']} positive tiles, "
            f"max confidence {s['max_confidence'] * 100:.0f}%"
        )
    elif not s["has_detection"]:
        reason = "no spill detected"
    else:
        reason = (
            f"below threshold ({s['positive_tiles']} tiles / "
            f"{s['max_confidence'] * 100:.0f}% conf)"
        )
    return AlertDecision(
        significant=significant,
        severity=severity_mod.classify_severity(
            s["max_confidence"], s["positive_tiles"], s["area_km2"]
        ),
        max_confidence=s["max_confidence"],
        positive_tiles=s["positive_tiles"],
        area_km2=s["area_km2"],
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def _region_label(investigation) -> str:
    return (getattr(investigation, "name", None) or "Unnamed area").strip()


def _detection_time(investigation, ml_result: dict | None) -> str:
    raw = (ml_result or {}).get("inference_timestamp")
    tiles = (ml_result or {}).get("tile_results") or []
    acq = next(
        (t.get("acquisition_date") for t in tiles if isinstance(t, dict) and t.get("acquisition_date")),
        None,
    )
    return acq or getattr(investigation, "end_date", None) or (raw or "unknown")


def _investigation_url(investigation_id: str) -> str:
    base = (settings.PUBLIC_BASE_URL or "").rstrip("/")
    return f"{base}/officer/incidents/{investigation_id}"


def _build_body(decision: AlertDecision, region: str, det_time: str, url: str) -> str:
    area = f"{decision.area_km2:,.0f} km²" if decision.area_km2 else "area n/a"
    return (
        f"ORICEAN ALERT — {decision.severity.upper()} oil spill detected\n"
        f"{region}\n"
        f"Detected: {det_time}\n"
        f"Confidence: {decision.max_confidence * 100:.0f}%  ·  "
        f"Tiles: {decision.positive_tiles}  ·  ~{area}\n"
        f"{url}"
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def _record_alert(
    db: AsyncSession,
    *,
    investigation_id: str | None,
    severity: str,
    headline: str,
    region_label: str | None,
    detection_time: str | None,
    decision: AlertDecision | None,
    url: str | None,
    recipients: list[str],
    body: str,
) -> Alert:
    """Send ``body`` to every recipient and persist one aggregate Alert row."""
    sent = 0
    details: list[str] = []
    for to in recipients:
        ok, detail = _send_sms(to, body)
        details.append(f"{to}: {'ok' if ok else detail}")
        if ok:
            sent += 1

    if not recipients:
        status = "skipped"
        detail_text = "no recipients configured"
    elif sent == len(recipients):
        status = "sent"
        detail_text = "; ".join(details)
    elif sent == 0:
        status = "not_configured" if details and "twilio_not_configured" in details[0] else "failed"
        detail_text = "; ".join(details)
    else:
        status = "partial"
        detail_text = "; ".join(details)

    alert = Alert(
        id=str(uuid.uuid4()),
        investigation_id=investigation_id,
        severity=severity,
        headline=headline,
        region_label=region_label,
        detection_time=detection_time,
        confidence=decision.max_confidence if decision else None,
        positive_tiles=decision.positive_tiles if decision else None,
        area_km2=decision.area_km2 if decision else None,
        investigation_url=url,
        delivery_status=status,
        delivery_detail=detail_text[:1000],
        recipients_count=len(recipients),
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    logger.info(
        "Alert recorded | id=%s severity=%s status=%s recipients=%d",
        alert.id, severity, status, len(recipients),
    )
    return alert


async def maybe_create_and_send_alert(
    db: AsyncSession,
    investigation,
    ml_result: dict | None,
) -> Alert | None:
    """Create + deliver an alert for a significant detection. Best-effort.

    No-op (returns ``None``) when: alerting disabled, detection not significant,
    or an alert already exists for this investigation.
    """
    try:
        cfg = await get_or_create_settings(db)
        if not cfg.enabled:
            return None

        existing = await db.execute(
            select(Alert.id).where(
                Alert.investigation_id == investigation.id, Alert.severity != "test"
            )
        )
        if existing.first() is not None:
            return None

        decision = evaluate_detection(
            ml_result, cfg.min_confidence, cfg.min_positive_tiles
        )
        if not decision.significant:
            logger.info(
                "No alert for investigation %s — %s", investigation.id, decision.reason
            )
            return None

        region = _region_label(investigation)
        det_time = _detection_time(investigation, ml_result)
        url = _investigation_url(investigation.id)
        recipients = settings_recipients(cfg)
        body = _build_body(decision, region, det_time, url)

        return await _record_alert(
            db,
            investigation_id=investigation.id,
            severity=decision.severity,
            headline=f"{decision.severity.capitalize()} oil spill detected — {region}",
            region_label=region,
            detection_time=det_time,
            decision=decision,
            url=url,
            recipients=recipients,
            body=body,
        )
    except Exception:  # pragma: no cover - alerting must never break inference
        logger.exception("maybe_create_and_send_alert failed (ignored)")
        return None


async def send_test_alert(db: AsyncSession) -> Alert:
    """Send a fixed test message to the configured recipients and record it."""
    cfg = await get_or_create_settings(db)
    recipients = settings_recipients(cfg)
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = (
        "ORICEAN ALERT — TEST\n"
        "This is a test of the officer notification channel.\n"
        f"Sent: {now}\n"
        "No action required."
    )
    return await _record_alert(
        db,
        investigation_id=None,
        severity="test",
        headline="Test notification",
        region_label=None,
        detection_time=now,
        decision=None,
        url=None,
        recipients=recipients,
        body=body,
    )
