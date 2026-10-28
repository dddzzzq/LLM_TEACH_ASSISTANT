export const ASSISTANT_SIZE = 52;
export const ASSISTANT_MARGIN = 16;
export const DRAG_THRESHOLD = 6;

export function clampAssistantPosition(position, viewport) {
  const maxX = Math.max(0, viewport.width - ASSISTANT_SIZE);
  const maxY = Math.max(0, viewport.height - ASSISTANT_SIZE);
  const marginX = Math.min(ASSISTANT_MARGIN, maxX / 2);
  const marginY = Math.min(ASSISTANT_MARGIN, maxY / 2);
  return {
    x: Math.min(maxX - marginX, Math.max(marginX, Number(position?.x) || 0)),
    y: Math.min(maxY - marginY, Math.max(marginY, Number(position?.y) || 0)),
  };
}

export function defaultAssistantPosition(viewport) {
  return clampAssistantPosition(
    { x: viewport.width - ASSISTANT_SIZE - 24, y: viewport.height - ASSISTANT_SIZE - 24 },
    viewport
  );
}

export function isAssistantDrag(start, point) {
  return Math.hypot(point.x - start.x, point.y - start.y) >= DRAG_THRESHOLD;
}

export function assistantPanelRect(position, viewport, expanded = false) {
  const margin = Math.min(12, viewport.width / 4, viewport.height / 4);
  const mobile = viewport.width <= 640;
  const width = Math.max(0, Math.min(expanded ? 1100 : 456, viewport.width - margin * 2));
  const height = Math.max(
    0,
    Math.min(expanded ? 860 : 680, viewport.height - (mobile || expanded ? margin * 2 : 88))
  );
  const preferredLeft =
    position.x + ASSISTANT_SIZE / 2 > viewport.width / 2
      ? position.x + ASSISTANT_SIZE - width
      : position.x;
  const preferredTop = position.y - height - 12;
  return {
    width,
    height,
    left:
      expanded || mobile
        ? (viewport.width - width) / 2
        : Math.min(viewport.width - width - margin, Math.max(margin, preferredLeft)),
    top:
      expanded || mobile
        ? margin
        : Math.min(viewport.height - height - margin, Math.max(margin, preferredTop)),
  };
}

export function assistantPositionKey(user) {
  return `teacher-assistant-position:${user?.user_id || user?.id || user?.username || 'teacher'}`;
}

export function assistantSessionKey(user) {
  return `teacher-assistant-session:${user?.user_id || user?.id || user?.username || 'teacher'}`;
}

export function readAssistantSession(storage, user) {
  try {
    return storage.getItem(assistantSessionKey(user)) || storage.getItem('current_session_id');
  } catch {
    return null;
  }
}

// Call only after the server has created or authorized the session for this account.
export function rememberAssistantSession(storage, user, sessionId) {
  try {
    storage.setItem(assistantSessionKey(user), sessionId);
    if (storage.getItem('current_session_id') === sessionId)
      storage.removeItem('current_session_id');
  } catch {
    // Persistent preference storage is optional; the in-memory session still works.
  }
}

export function forgetAssistantSession(storage, user, sessionId) {
  try {
    if (storage.getItem(assistantSessionKey(user)) === sessionId)
      storage.removeItem(assistantSessionKey(user));
    if (storage.getItem('current_session_id') === sessionId)
      storage.removeItem('current_session_id');
  } catch {
    // Storage may be disabled without preventing a new conversation.
  }
}
