export const UNASSIGNED_CLASS = '__unassigned__';
export const classCanImport = (item) => item?.source === 'portal';
export function canImportClass(assignment, item) {
  if (!assignment || !item) return false;
  if (item.can_import) return true;
  if (item.source === 'unassigned')
    return ['', '全部班级', '所有班级', '未分班', '未分班材料', '历史未分班成绩'].includes(
      (assignment.class_name || '').trim()
    );
  return (
    !assignment.portal_homework_id && item.source === 'local' && item.name === assignment.class_name
  );
}
export function chooseClass(classes, requested, remembered) {
  return (
    classes.find((item) => item.id === requested)?.id ||
    classes.find((item) => item.id === remembered)?.id ||
    classes[0]?.id ||
    ''
  );
}
export function finalScore(result) {
  return result.is_human_reviewed && Number.isFinite(result.human_score)
    ? result.human_score
    : result.score;
}
export function filterClassResults(results, search = '', review = '') {
  const query = search.trim().toLocaleLowerCase();
  return results
    .filter(
      (result) =>
        (!query ||
          [result.student_id, result.student_name].some((value) =>
            String(value || '')
              .toLocaleLowerCase()
              .includes(query)
          )) &&
        (!review || (review === 'reviewed' ? result.is_human_reviewed : !result.is_human_reviewed))
    )
    .sort((a, b) => finalScore(b) - finalScore(a));
}
