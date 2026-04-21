// k6 load-test script for the I3 HTTP surface.
//
// Matches the Locust scenario in ../locustfile.py:
//   - 10 virtual users
//   - 30 s ramp-up
//   - 5 min steady
//   - 30 s ramp-down
//
// Usage:
//   k6 run benchmarks/k6/load.js --env HOST=http://localhost:8000
//
// The script exercises the REST endpoints only; the WebSocket flow is
// covered by locust in our reference setup, but a parallel k6 ws
// scenario can be added under the same `benchmarks/k6/` folder.

import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep, group } from 'k6';
import { Trend, Counter } from 'k6/metrics';

const HOST = __ENV.HOST || 'http://localhost:8000';

const wsLatency = new Trend('ws_message_latency_ms');
const restErrors = new Counter('rest_errors');

export const options = {
  stages: [
    { duration: '30s', target: 10 },  // ramp up
    { duration: '5m',  target: 10 },  // steady
    { duration: '30s', target: 0  },  // ramp down
  ],
  thresholds: {
    'http_req_failed':   ['rate<0.01'],
    'http_req_duration': ['p(95)<250', 'p(99)<500'],
    'ws_message_latency_ms': ['p(95)<400'],
  },
};

function randomUserId() {
  return 'k6-' + Math.random().toString(16).slice(2, 10);
}

export default function () {
  const userId = randomUserId();

  group('rest', function () {
    const health = http.get(`${HOST}/api/health`, { tags: { name: 'GET /api/health' } });
    if (!check(health, { 'health 200': (r) => r.status === 200 })) {
      restErrors.add(1);
    }

    const seed = http.post(
      `${HOST}/api/demo/seed`,
      JSON.stringify({ user_id: userId, messages: 3 }),
      { headers: { 'Content-Type': 'application/json' }, tags: { name: 'POST /api/demo/seed' } },
    );
    if (!check(seed, { 'seed 200': (r) => r.status === 200 || r.status === 201 })) {
      restErrors.add(1);
    }

    const profile = http.get(`${HOST}/api/user/${userId}/profile`, {
      tags: { name: 'GET /api/user/[id]/profile' },
    });
    check(profile, { 'profile ok': (r) => r.status === 200 || r.status === 404 });
  });

  sleep(1 + Math.random());
}

// Optional WebSocket scenario (opt-in via --tag scenario=ws).
export function wsScenario() {
  const userId = randomUserId();
  const wsHost = HOST.replace(/^http/, 'ws');
  const url = `${wsHost}/ws/${userId}`;

  const res = ws.connect(url, {}, function (socket) {
    socket.on('open', function () {
      const t0 = Date.now();
      socket.send(JSON.stringify({ type: 'message', text: 'hi', timestamp_ms: t0 }));
      socket.on('message', function (raw) {
        try {
          const frame = JSON.parse(raw);
          if (frame.type === 'response') {
            wsLatency.add(Date.now() - t0);
            socket.close();
          }
        } catch (_err) {
          // non-JSON frames are ignored
        }
      });
      socket.setTimeout(function () { socket.close(); }, 10_000);
    });
  });
  check(res, { 'ws connected': (r) => r && r.status === 101 });
}
