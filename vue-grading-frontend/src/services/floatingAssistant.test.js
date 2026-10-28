import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ASSISTANT_SIZE,
  clampAssistantPosition,
  defaultAssistantPosition,
  isAssistantDrag,
  assistantPanelRect,
  assistantPositionKey,
  assistantSessionKey,
  readAssistantSession,
  rememberAssistantSession,
  forgetAssistantSession,
} from './floatingAssistant.js';

test('drag threshold distinguishes a click from moving the assistant', () => {
  assert.equal(isAssistantDrag({ x: 20, y: 20 }, { x: 23, y: 23 }), false);
  assert.equal(isAssistantDrag({ x: 20, y: 20 }, { x: 26, y: 20 }), true);
});

test('saved coordinates stay reachable after a desktop to phone resize', () => {
  const desktop = defaultAssistantPosition({ width: 1920, height: 1080 });
  const phone = clampAssistantPosition(desktop, { width: 390, height: 400 });
  assert.ok(phone.x >= 0 && phone.x + ASSISTANT_SIZE <= 390);
  assert.ok(phone.y >= 0 && phone.y + ASSISTANT_SIZE <= 400);
  const narrow = clampAssistantPosition({ x: -40, y: Infinity }, { width: 40, height: 45 });
  assert.ok(Number.isFinite(narrow.x) && Number.isFinite(narrow.y));
});

test('floating and expanded panels stay inside either edge and small viewports', () => {
  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 390, height: 844 },
    { width: 390, height: 320 },
  ]) {
    for (const position of [{ x: 16, y: 16 }, defaultAssistantPosition(viewport)]) {
      for (const expanded of [false, true]) {
        const box = assistantPanelRect(position, viewport, expanded);
        assert.ok(box.left >= 0 && box.top >= 0);
        assert.ok(box.left + box.width <= viewport.width);
        assert.ok(box.top + box.height <= viewport.height);
      }
    }
  }
});

test('assistant position is scoped to the signed-in account', () => {
  assert.notEqual(assistantPositionKey({ user_id: 1 }), assistantPositionKey({ user_id: 2 }));
  assert.equal(assistantPositionKey({ id: 7 }), assistantPositionKey({ user_id: 7 }));
});

test('verified legacy sessions migrate without being reused by a different account', () => {
  const data = new Map([['current_session_id', 'session-a']]);
  const storage = {
    getItem: (key) => data.get(key) || null,
    setItem: (key, value) => data.set(key, value),
    removeItem: (key) => data.delete(key),
  };
  const a = { user_id: 1 },
    b = { user_id: 2 };
  assert.equal(readAssistantSession(storage, a), 'session-a');
  rememberAssistantSession(storage, a, 'session-a');
  assert.equal(data.has('current_session_id'), false);
  assert.equal(readAssistantSession(storage, b), null);
  rememberAssistantSession(storage, b, 'session-b');
  assert.equal(readAssistantSession(storage, a), 'session-a');
  assert.equal(readAssistantSession(storage, b), 'session-b');
  forgetAssistantSession(storage, b, 'session-a');
  assert.equal(data.get(assistantSessionKey(a)), 'session-a');
  assert.equal(data.get(assistantSessionKey(b)), 'session-b');
  forgetAssistantSession(storage, a, 'session-a');
  assert.equal(readAssistantSession(storage, a), null);
});

test('blocked preference storage does not prevent an in-memory chat', () => {
  const storage = {
    getItem() {
      throw new Error('blocked');
    },
    setItem() {
      throw new Error('blocked');
    },
    removeItem() {
      throw new Error('blocked');
    },
  };
  assert.equal(readAssistantSession(storage, { user_id: 1 }), null);
  assert.doesNotThrow(() => rememberAssistantSession(storage, { user_id: 1 }, 'session'));
  assert.doesNotThrow(() => forgetAssistantSession(storage, { user_id: 1 }, 'session'));
});
