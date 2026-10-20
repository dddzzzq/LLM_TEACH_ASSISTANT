"""Portal export options reconstructed from the actual failed task diagnostics."""

PORTAL_EXPORT_OPTIONS = '''
<h3>导出范围</h3>
<div class=export-range><span class="grade_check grade_checked" onclick="selectExportOption(this,'.export-range')"></span>导出本班级作业附件</div>
<div class="export-range all"><span class=grade_check onclick="selectExportOption(this,'.export-range')"></span>导出作业下所有班级作业附件
<p>若班级内没有学生提交，将自动跳过导出</p></div>
<h3>导出格式</h3>
<div class="out export-flexBox"><span class="grade_check grade_checked" onclick="selectExportOption(this,'.out')"></span>导出完整答题记录
<p>导出内容包含学生作答情况、教师批改情况、学生提交附件</p></div>
<div class="out export-flexBox"><span class=grade_check onclick="selectExportOption(this,'.out')"></span>仅导出文档留痕批注</div>
<div class="out export-flexBox" id=attachment-option><span class=grade_check onclick="selectExportOption(this,'.out')"></span>导出提交附件
<p>以压缩包格式导出学生提交的附件</p></div>
<h3>命名格式</h3>
<div class="nameItem clearfix" id=default-naming><span class="grade_check grade_checked" onclick="selectExportOption(this,'.nameItem')"></span>默认格式
<p>学生提交的文件将保留文件原名；答题记录按照“院系-专业-班级-学号-姓名”命名</p></div>
<div class="nameItem clearfix"><span class=grade_check onclick="selectExportOption(this,'.nameItem')"></span>自定义</div>
<style>.grade_check{display:inline-block;width:20px;height:20px}</style>
<script>
function selectExportOption(element, group) {
  if(group==='.nameItem') window.namingClicks=(window.namingClicks||0)+1;
  document.querySelectorAll(group+' .grade_check').forEach(x=>x.classList.remove('grade_checked'));
  element.classList.add('grade_checked');
}
</script>
'''
