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
    file(id, path) {
      return client.get(`${base}/${id}/file`, { params: { path }, responseType: 'blob' });
    },
    retry(id) {
      return client.post(`${base}/${id}/retry`);
    },
    grade(id, allowPartial) {
      return client.post(`${base}/${id}/grade`, { allow_partial: allowPartial });
    },
    regrade(id, allowPartial, previousJobId, idempotencyKey) {
      return client.post(
        `${base}/${id}/grade`,
        { allow_partial: allowPartial, regrade: true, previous_job_id: previousJobId },
        { headers: { 'Idempotency-Key': idempotencyKey } }
      );
    },
    cancel(id) {
      return client.post(`${base}/${id}/cancel`);
    },
  };
}
