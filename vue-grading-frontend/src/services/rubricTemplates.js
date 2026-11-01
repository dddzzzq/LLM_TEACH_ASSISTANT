// Templates fill an assignment's rubric draft. Only criteria are saved, so each
// assignment and each grading snapshot keeps its own independent scoring rules.
export const rubricTemplates = [
  {
    id: 'problem_solving',
    name: '答题解题类',
    description:
      '适用于简答、计算、证明和推导。以题目、小题和关键步骤为依据，分别评价答案、过程和概念运用。',
    criteria: [
      {
        name: '答案正确性',
        score: 50,
        description:
          '按各小题核对答案、计算结果和最终结论，优先采用题目明确的小题权重。答案正确且覆盖全部问题得满分；局部计算、结论错误仅扣对应部分，漏答不得分，不将同一错误重复扣到其他正确小题。',
      },
      {
        name: '解题过程与逻辑',
        score: 30,
        description:
          '核对解题方法、推导或证明中的关键步骤、依据和逻辑衔接。思路正确、步骤充分得满分；关键步骤缺失或推理不成立按影响扣分，计算失误但方法正确时保留过程分，允许合理的等价解法。',
      },
      {
        name: '知识点运用',
        score: 15,
        description:
          '核对概念、定理、公式及其适用条件，能正确对应题目中的已知条件与待求量。概念混淆、公式误用或忽略必要条件按实际影响扣分；不因学生采用不同于参考答案的正确知识路径扣分。',
      },
      {
        name: '表达与规范',
        score: 5,
        description:
          '题号与作答对应，符号定义、单位、图表和文字说明清楚，能够核查答案与过程。按影响理解的实际问题扣分；不以篇幅长短、未要求的排版或个人书写风格代替内容评价。',
      },
    ],
  },
  {
    id: 'engineering_code',
    name: '工程代码类',
    description:
      '适用于编程、算法实现和软件项目。围绕需求、代码实现、边界处理和验证材料评分，不将代码篇幅作为完成度。',
    criteria: [
      {
        name: '功能与需求完成度',
        score: 45,
        description:
          '按题目逐项核对必需功能、输入输出、接口和约束。核心流程完整、结果符合要求得满分；功能缺失、输出错误或实现偏离需求按对应功能的重要性扣分，未要求的附加功能不替代必需功能得分。',
      },
      {
        name: '算法与结构设计',
        score: 20,
        description:
          '核对算法、数据结构、模块划分及调用关系，评价实现逻辑是否正确、职责是否清楚。明显的逻辑缺陷、重复耦合和不合理设计按影响扣分；时间或空间复杂度要求以题目明确约束为准，不强制使用未指定的框架或设计模式。',
      },
      {
        name: '边界与异常处理',
        score: 15,
        description:
          '核对与任务相关的空输入、极值、非法输入和失败路径，查看是否有必要的校验、错误提示及资源处理。根据代码或验证材料中可核查的遗漏扣分；不要求处理与题目无关的异常，不臆测缺陷。',
      },
      {
        name: '测试与结果验证',
        score: 10,
        description:
          '核对测试用例、运行结果或其他验证材料是否覆盖主要功能和关键边界，结果能否支撑完成结论。题目要求测试材料时按覆盖程度和可信证据给分；未要求单独报告时可结合已提交的验证内容评价，不因缺少测试报告直接扣分，未执行代码不得声称测试通过。',
      },
      {
        name: '代码规范与说明',
        score: 10,
        description:
          '代码命名、文件组织和必要注释清楚，依赖、入口与使用方式可以理解；题目要求的源码、配置和说明齐全。只按实际影响阅读、运行或核查的问题扣分，未要求的文档不额外设为必交材料。',
      },
    ],
  },
  {
    id: 'research_paper',
    name: '论文研究类',
    description:
      '适用于课程论文、文献综述和调研报告。重点评价研究问题、方法与论证、证据和引用，不套用编程项目要求。',
    criteria: [
      {
        name: '研究问题与任务契合',
        score: 20,
        description:
          '研究主题、问题、范围及目标明确，回应题目要求并交代研究背景。问题清楚且任务覆盖完整得满分；偏题、范围失控或核心问题缺失按程度扣分，不以选题是否热门评价质量。',
      },
      {
        name: '分析与论证',
        score: 30,
        description:
          '论点清晰，研究或分析方法适合问题，分析过程能够支持结论并说明必要的局限。文献综述评价比较、归纳与评述，调研报告评价调查与分析过程；资料堆砌、逻辑跳跃和无依据结论按影响扣分，不强制所有论文包含实验或代码。',
      },
      {
        name: '证据与文献引用',
        score: 25,
        description:
          '关键论点有相关文献、数据或案例支撑，文内引用与参考文献对应，来源信息可核查。证据不足、引用遗漏或结论超出证据按实际问题扣分；无法核验的来源应指出待核查，不无依据认定伪造，不自行规定文献数量或年份。',
      },
      {
        name: '结构与学术表达',
        score: 15,
        description:
          '章节层次和段落衔接合理，概念与术语准确，图表服务于论证，结论回应研究问题。结构混乱、表述歧义或结论与正文矛盾按影响扣分；不以篇幅长短或措辞华丽代替研究质量。',
      },
      {
        name: '格式与提交规范',
        score: 10,
        description:
          '核对题目明确规定的篇幅、章节、图表标注、引文体例和提交格式，按实际缺失或不一致给分。题目未规定的排版、引文样式或封面要求不另设扣分项。',
      },
    ],
  },
  {
    id: 'presentation',
    name: '展示汇报类',
    description:
      '适用于幻灯片、海报和书面汇报。评价提交材料中的内容、叙述、证据和视觉呈现；PPT 请先导出为 PDF，现场表达与答辩按实际记录另行设置。',
    criteria: [
      {
        name: '内容准确与任务完成',
        score: 30,
        description:
          '核对汇报主题、核心概念、工作成果和结论是否准确，是否覆盖题目要求。内容准确且任务完整得满分；关键事实错误、成果表述失实或要求遗漏按影响扣分，不以页数或动画数量代替内容质量。',
      },
      {
        name: '结构与叙述逻辑',
        score: 25,
        description:
          '背景、问题、方案或过程、成果与结论组织清楚，标题和页面顺序能引导受众理解。叙述跳跃、前后矛盾和缺少关键衔接按影响扣分，允许适合主题的不同汇报结构。',
      },
      {
        name: '重点提炼与证据',
        score: 25,
        description:
          '核心观点、工作贡献和关键结果突出，有相关数据、案例、对比或演示材料支撑，图表能够说明结论。信息堆砌、重点不明或证据不足按影响扣分；不因未提交题目未要求的代码或实验材料扣分。',
      },
      {
        name: '视觉呈现与规范',
        score: 20,
        description:
          '根据实际可见材料评价版面层次、字号对比、图表可读性、标注与来源，并核对提交要求。只根据可见问题扣分；无法查看的原始版式不臆测评价，未提供现场记录时不评价语速、台风、互动或答辩表现。',
      },
    ],
  },
];

// Editing a draft must never mutate the templates used by another assignment.
for (const template of rubricTemplates) {
  template.criteria.forEach(Object.freeze);
  Object.freeze(template.criteria);
  Object.freeze(template);
}
Object.freeze(rubricTemplates);

export const defaultRubricTemplateId = 'problem_solving';

export function createRubricFromTemplate(id = defaultRubricTemplateId) {
  const template = rubricTemplates.find((item) => item.id === id);
  if (!template) throw new Error('评分模板不存在');
  return JSON.stringify(
    Object.fromEntries(
      template.criteria.map(({ name, score, description }) => [name, { score, description }])
    )
  );
}

export function matchingRubricTemplate(value) {
  try {
    const rubric = typeof value === 'string' ? JSON.parse(value) : value;
    if (!rubric || typeof rubric !== 'object' || Array.isArray(rubric)) return null;
    return (
      rubricTemplates.find(
        (template) =>
          Object.keys(rubric).length === template.criteria.length &&
          template.criteria.every(({ name, score, description }) => {
            const criterion = rubric[name];
            return (
              criterion &&
              typeof criterion === 'object' &&
              !Array.isArray(criterion) &&
              Object.keys(criterion).length === 2 &&
              criterion.score === score &&
              criterion.description === description
            );
          })
      ) || null
    );
  } catch {
    return null;
  }
}
