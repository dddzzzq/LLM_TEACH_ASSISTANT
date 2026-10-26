export function createPortalApi(client) {
  const base = '/api/portal';
  const id = encodeURIComponent;
  return {
    connection: () => client.get(`${base}/connection`),
    save: (body) => client.put(`${base}/connection`, body),
    disconnect: () => client.delete(`${base}/connection`),
    reauth: () => client.post(`${base}/connection/reauth`),
    sync: () => client.post(`${base}/catalog-syncs`),
    courses: () => client.get(`${base}/courses`),
    homeworks: (courseId) => client.get(`${base}/courses/${id(courseId)}/homeworks`),
    homeworkDetail: (homeworkId) => client.get(`${base}/homeworks/${id(homeworkId)}/detail`),
    readHomeworkDetail: (homeworkId, snapshotId) =>
      client.post(`${base}/homeworks/${id(homeworkId)}/detail-reads`, { snapshot_id: snapshotId }),
    homeworkDetailSource: (homeworkId, sourceId) =>
      client.get(`${base}/homeworks/${id(homeworkId)}/detail/sources/${id(sourceId)}`, {
        responseType: 'blob',
      }),
    operation: (operationId) => client.get(`${base}/operations/${id(operationId)}`),
    cancelDetailRead: (operationId) => client.post(`${base}/operations/${id(operationId)}/cancel`),
    events: (operationId, after = 0) =>
      client.get(`${base}/operations/${id(operationId)}/events`, { params: { after } }),
    createWorkflow: (body, key) =>
      client.post('/api/grading/workflows', body, { headers: { 'Idempotency-Key': key } }),
    workflow: (workflowId) => client.get(`/api/grading/workflows/${id(workflowId)}`),
    cancelWorkflow: (workflowId) => client.post(`/api/grading/workflows/${id(workflowId)}/cancel`),
    retryWorkflow: (workflowId) => client.post(`/api/grading/workflows/${id(workflowId)}/retry`),
    legacyAssignments: () => client.get(`${base}/legacy-assignments`),
    claimAssignment: (assignmentId, ownerId) =>
      client.put(`${base}/legacy-assignments/${id(assignmentId)}`, { owner_id: ownerId }),
  };
}

export const operationActive = (op) =>
  op && !['SUCCEEDED', 'FAILED', 'CANCELLED', 'INTERRUPTED', 'EXPIRED'].includes(op.status);
export function portalStatus(value) {
  return (
    {
      UNCONFIGURED: '未配置',
      INITIALIZING: '正在连接',
      READY: '可用',
      REAUTH_REQUIRED: '需要认证',
      DISABLED: '已断开',
      QUEUED: '已排队',
      RUNNING: '执行中',
      WAITING_AUTH: '等待本人认证',
      WAITING_OTP: '等待验证码',
      WAITING_IDENTITY: '等待核对身份',
      WAITING_REVIEW: '等待核对材料',
      RETRY_QUEUED: '等待重试下载',
      SUCCEEDED: '已完成',
      PARTIAL_SUCCESS: '部分完成',
      FAILED: '失败',
      CANCELLED: '已取消',
      INTERRUPTED: '已中断',
      EXPIRED: '已超时',
    }[value] ||
    value ||
    '未知'
  );
}
