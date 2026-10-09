<template>
  <div class="home-view">
    <div class="page-heading">
      <div><h1 class="page-title">教学概览</h1><p class="page-description">欢迎回来，从这里开始今天的教学工作。</p></div>
      <div class="university-mark"><img :src="logo" alt="西安电子科技大学校徽" /><span>西安电子科技大学</span></div>
    </div>
    <section class="welcome-panel" aria-labelledby="welcome-title">
      <div class="welcome-copy">
        <span class="welcome-label"><AppIcon name="sparkles" /> 智能教学助理</span>
        <h2 id="welcome-title">让批改更高效，<br />让教学更专注。</h2>
        <p>以大语言模型辅助作业查重、智能评分与试卷批改，<br class="desktop-break" />把更多时间留给思考、交流与教学。</p>
        <router-link to="/assignments" class="primary-button">开始作业批改 <AppIcon name="arrow" /></router-link>
      </div>
      <div class="teaching-illustration" aria-hidden="true">
        <div class="illustration-orbit orbit-one"></div><div class="illustration-orbit orbit-two"></div>
        <div class="paper-sheet paper-back"></div>
        <div class="paper-sheet paper-front">
          <div class="paper-heading"><AppIcon name="book" /><span>教学 · 有所助益</span></div>
          <div class="paper-rule long"></div><div class="paper-rule short"></div>
          <div class="paper-check"><span>✓</span><div class="paper-rule"></div></div>
          <div class="paper-check"><span>✓</span><div class="paper-rule"></div></div>
          <div class="paper-check"><span>✓</span><div class="paper-rule"></div></div>
          <div class="paper-note">让每一份作业，都得到认真回应。</div>
        </div>
        <div class="illustration-badge"><AppIcon name="sparkles" /><span>AI 辅助 · 教师复核</span></div>
      </div>
    </section>
    <section class="feature-section" aria-labelledby="feature-title">
      <div class="section-heading"><h2 id="feature-title">教学工具</h2><span>选择一个工具，开始工作</span></div>
      <div class="feature-grid">
        <router-link to="/assignments" class="feature-card">
          <div class="feature-icon icon-teal"><AppIcon name="check" /></div>
          <h3>作业查重与评分</h3><p>管理课程作业，批量分析内容相似度，按评分标准生成成绩与评语。</p>
          <span class="feature-action">进入作业管理 <AppIcon name="arrow" /></span>
        </router-link>
        <router-link to="/exams" class="feature-card">
          <div class="feature-icon icon-blue"><AppIcon name="pen" /></div>
          <h3>主观题智能评分</h3><p>上传学生答卷，识别作答内容，查看逐题评分与完整试卷报告。</p>
          <span class="feature-action">进入试卷管理 <AppIcon name="arrow" /></span>
        </router-link>
        <router-link to="/ai-assistant" class="feature-card">
          <div class="feature-icon icon-amber"><AppIcon name="chat" /></div>
          <h3>AI 教学助手</h3><p>通过对话下载教务系统作业、接续批改，或查询学生成绩。</p>
          <span class="feature-action">开始对话 <AppIcon name="arrow" /></span>
        </router-link>
      </div>
    </section>
    <section class="workflow-panel" aria-labelledby="workflow-title">
      <div class="workflow-intro"><AppIcon name="book" /><div><h2 id="workflow-title">从提交到反馈</h2><p>有序完成每一次批改</p></div></div>
      <ol class="workflow-steps">
        <li><span>1</span><div><h3>创建任务</h3><p>设置题目与评分标准</p></div></li>
        <li><span>2</span><div><h3>上传作业</h3><p>提交作业或试卷材料</p></div></li>
        <li><span>3</span><div><h3>智能批改</h3><p>查看 AI 评分与分析</p></div></li>
        <li><span>4</span><div><h3>教师复核</h3><p>核对成绩并导出结果</p></div></li>
      </ol>
    </section>
    <p class="home-footnote"><AppIcon name="shield" /> AI 为教学提供辅助，最终评价由教师把关。</p>
  </div>
</template>

<script setup>
// 导入本地 logo 图片
import logo from "@/assets/logo.png";
import AppIcon from '../components/AppIcon.vue';
</script>

<style scoped>
.university-mark { display: flex; align-items: center; gap: 9px; color: #66788a; font-size: 12px; }
.university-mark img { width: 30px; height: 30px; object-fit: contain; }
.welcome-panel { position: relative; display: flex; justify-content: space-between; align-items: center; min-height: 326px; padding: 38px 44px; border: 1px solid #d5e8e5; border-radius: 18px; background: #eaf4f2; overflow: hidden; }
.welcome-copy { position: relative; z-index: 1; }
.welcome-label { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; color: #14655d; }
.welcome-label svg { width: 16px; height: 16px; }
.welcome-copy h2 { margin-top: 15px; color: #173d3b; font-size: clamp(30px, 2.7vw, 42px); font-weight: 700; line-height: 1.45; letter-spacing: -1.5px; }
.welcome-copy p { margin-top: 15px; color: #597875; font-size: 13px; line-height: 1.9; }
.welcome-copy .primary-button { margin-top: 24px; }
.teaching-illustration { position: relative; width: 345px; height: 280px; flex-shrink: 0; }
.illustration-orbit { position: absolute; border: 1px solid #cce2dd; border-radius: 50%; width: 260px; height: 260px; top: 10px; left: 50px; }
.orbit-two { width: 310px; height: 310px; top: -15px; left: 25px; }
.paper-sheet { position: absolute; width: 195px; height: 236px; top: 18px; left: 85px; border-radius: 10px; }
.paper-back { background: #c9e3dc; transform: rotate(12deg); border: 1px solid #bad9d0; }
.paper-front { background: #fff; transform: rotate(-7deg); padding: 23px 20px; box-shadow: 0 12px 26px rgb(39 95 81 / 0.1); border: 1px solid #dce9e5; }
.paper-heading { display: flex; align-items: center; gap: 8px; color: #197b70; font-size: 11px; font-weight: 600; margin-bottom: 21px; }
.paper-heading svg { width: 20px; height: 20px; }
.paper-rule { height: 5px; border-radius: 3px; background: #e8efee; flex: 1; }
.paper-rule.long { width: 90%; margin-bottom: 8px; }
.paper-rule.short { width: 60%; margin-bottom: 21px; }
.paper-check { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
.paper-check span { display: grid; place-items: center; width: 17px; height: 17px; background: #e4f3ed; border-radius: 4px; color: #278e82; font-size: 11px; }
.paper-note { color: #66788a; font-size: 8px; margin-top: 18px; }
.illustration-badge { position: absolute; bottom: 13px; left: 35px; display: flex; align-items: center; gap: 9px; background: #fff; border: 1px solid #dce9e5; border-radius: 9px; padding: 13px 17px; color: #344b5f; font-size: 11px; box-shadow: 0 5px 15px rgb(39 95 81 / 0.05); }
.illustration-badge svg { width: 19px; height: 19px; color: #197b70; }
.feature-section { margin-top: 30px; }
.section-heading { display: flex; align-items: center; justify-content: space-between; margin-bottom: 17px; }
.section-heading h2 { font-size: 17px; font-weight: 700; }
.section-heading > span { font-size: 12px; color: #66788a; }
.feature-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }
.feature-card { display: flex; flex-direction: column; align-items: flex-start; padding: 25px; background: #fff; border: 1px solid #e0e7ee; border-radius: 13px; transition: border-color 150ms, box-shadow 150ms; }
.feature-card:hover { border-color: #83c4b9; box-shadow: 0 4px 16px rgb(20 45 64 / 0.04); }
.feature-icon { display: grid; place-items: center; width: 44px; height: 44px; border-radius: 11px; margin-bottom: 20px; }
.feature-icon svg { width: 23px; height: 23px; }
.icon-teal { background: #edf7f5; color: #197b70; }
.icon-blue { background: #edf3fc; color: #4a73b2; }
.icon-amber { background: #fcf4e8; color: #af7b2f; }
.feature-card h3 { font-size: 16px; font-weight: 700; }
.feature-card p { font-size: 12px; line-height: 1.9; color: #66788a; margin: 10px 0 23px; max-width: 27em; }
.feature-action { display: flex; align-items: center; gap: 8px; margin-top: auto; font-size: 12px; font-weight: 600; color: #197b70; }
.feature-action svg { width: 16px; height: 16px; }
.workflow-panel { display: flex; align-items: center; gap: 25px; padding: 25px 27px; margin-top: 24px; background: #fff; border: 1px solid #e0e7ee; border-radius: 13px; }
.workflow-intro { display: flex; align-items: center; gap: 12px; padding-right: 25px; border-right: 1px solid #e0e7ee; flex-shrink: 0; }
.workflow-intro > svg { width: 25px; height: 25px; color: #66788a; }
.workflow-intro h2 { font-size: 13px; font-weight: 600; }
.workflow-intro p { font-size: 11px; color: #66788a; margin-top: 6px; }
.workflow-steps { display: grid; grid-template-columns: repeat(4, 1fr); flex: 1; gap: 15px; }
.workflow-steps li { display: flex; align-items: center; gap: 10px; }
.workflow-steps li > span { display: grid; place-items: center; height: 25px; width: 25px; flex-shrink: 0; border: 1px solid #d9e7e4; color: #197b70; border-radius: 50%; font-size: 11px; }
.workflow-steps h3 { font-size: 12px; font-weight: 600; }
.workflow-steps p { color: #66788a; font-size: 10px; margin-top: 6px; }
.home-footnote { display: flex; justify-content: center; align-items: center; gap: 6px; margin-top: 24px; color: #66788a; font-size: 11px; }
.home-footnote svg { width: 14px; height: 14px; }
@media (max-width: 1200px) {
  .welcome-panel { padding: 32px; }
  .teaching-illustration { width: 280px; transform: scale(.9); transform-origin: right center; }
  .workflow-panel { flex-direction: column; align-items: stretch; gap: 20px; }
  .workflow-intro { border-right: 0; padding-right: 0; }
}
@media (max-width: 960px) and (min-width: 761px) {
  .teaching-illustration { display: none; }
  .feature-grid { grid-template-columns: 1fr; }
}
@media (max-width: 760px) {
  .university-mark { font-size: 10px; }
  .university-mark img { width: 24px; height: 24px; }
  .welcome-panel { padding: 28px 24px; min-height: 320px; }
  .welcome-copy h2 { font-size: 31px; }
  .teaching-illustration { display: none; }
  .welcome-copy p { font-size: 12px; }
  .desktop-break { display: none; }
  .section-heading > span { font-size: 10px; }
  .feature-grid { grid-template-columns: 1fr; gap: 12px; }
  .feature-card { padding: 22px; }
  .feature-icon { margin-bottom: 16px; }
  .feature-card p { max-width: none; margin-bottom: 18px; }
  .workflow-panel { padding: 23px; }
  .workflow-steps { grid-template-columns: repeat(2, 1fr); gap: 22px 14px; }
}
</style>
