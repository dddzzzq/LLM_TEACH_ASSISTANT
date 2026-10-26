// The authenticated client is supplied by the caller; this module only owns the HTTP contract.
export function createMaterialsApi(client) {
  const base = '/api/materials';
  return {
    list({ assignmentId, sourceJobId, classId } = {}) {
      return client.get(base, {
        params: {
          assignment_id: assignmentId || undefined,
          source_job_id: sourceJobId || undefined,
          ...(classId ? { class_id: classId } : {}),
        },
      });
    },
    create(body, idempotencyKey) {
      return client.post(base, body, { headers: { 'Idempotency-Key': idempotencyKey } });
    },
    get(id) {
      return client.get(`${base}/${id}`);
    },
    retry(id) {
      return client.post(`${base}/${id}/retry`);
    },
    grade(id, allowPartial) {
      return client.post(`${base}/${id}/grade`, { allow_partial: allowPartial });
    },
    cancel(id) {
      return client.post(`${base}/${id}/cancel`);
    },
  };
}
