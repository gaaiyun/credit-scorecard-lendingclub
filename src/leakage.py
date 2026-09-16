# -*- coding: utf-8 -*-
"""字段泄漏判定：Lending Club accepted 表 151 列，逐列判「申请时点能不能拿到」。

**为什么用白名单而不是黑名单**
黑名单一定会漏。Lending Club 这份数据里放款后才产生的字段有 40 多个，
漏掉任何一个（`last_fico_range_high`、`settlement_amount`、`hardship_dpd`……）
模型 KS 就能冲到 0.9 以上，上线即失效。所以这里反过来做：
默认全部剔除，只有明确判定为「申请时点可得」的才进候选池，每一列都写理由。

分类口径
--------
MODEL    申请时点可得（申请信息 + 征信报告字段），进候选池
PRICING  Lending Club 自己的定价输出（grade / sub_grade / int_rate / installment）。
         申请人在签约前确实看得到，但它们是 LC 内部评分模型的**输出**，
         放进模型等于在学别人的模型。主卡剔除，对照卡可用，int_rate 留给收益测算。
LEAK     放款后才产生，泄漏，必须剔除
TARGET   用于构造标签或样本划分，不进模型
ID       标识列，无预测意义
TEXT     自由文本，本项目不做 NLP
SPARSE   联合申请 / 第二申请人字段，绝大多数样本为空
PLATFORM 平台内部字段，与借款人风险无因果关系，新申请人身上也不存在
"""
from __future__ import annotations

import pandas as pd

# (列名, 分类, 理由) —— 顺序与原始 CSV 的 151 列一致
FIELDS: dict[str, tuple[str, str]] = {
    # ---- 标识 ----
    "id":                 ("ID", "贷款流水号"),
    "member_id":          ("ID", "会员号，该版本已全部脱敏为空"),

    # ---- 申请信息 ----
    "loan_amnt":          ("MODEL", "申请金额，申请人自己填的"),
    "funded_amnt":        ("LEAK", "实际放款金额，放款后才确定；申请时点只有 loan_amnt"),
    "funded_amnt_inv":    ("LEAK", "投资人实际认购额，放款后由平台撮合结果决定"),
    "term":               ("MODEL", "期限，申请时选定。本项目只建 36 期的卡，故建模时为常数"),
    "int_rate":           ("PRICING", "LC 定价利率，是其内部评分的输出；留给收益测算算利息收入"),
    "installment":        ("PRICING", "月供 = f(金额, 期限, 利率)，含利率信息"),
    "grade":              ("PRICING", "LC 信用等级，其内部模型的输出"),
    "sub_grade":          ("PRICING", "LC 细分等级，同上"),
    "emp_title":          ("TEXT", "职位自由文本，30 万+ 不同取值，本项目不做 NLP"),
    "emp_length":         ("MODEL", "工作年限，申请时填报"),
    "home_ownership":     ("MODEL", "住房状况，申请时填报"),
    "annual_inc":         ("MODEL", "年收入，申请时填报"),
    "verification_status": ("MODEL", "收入是否经过核验，申请环节产生"),
    "issue_d":            ("TARGET", "放款月份，用于 vintage 分组与跨时间切分"),
    "loan_status":        ("TARGET", "贷款状态，标签来源"),
    "pymnt_plan":         ("LEAK", "是否进入还款计划，放款后状态"),
    "url":                ("ID", "LC 网页链接"),
    "desc":               ("TEXT", "借款描述自由文本，2013 年后 LC 已停止采集，缺失率极高"),
    "purpose":            ("MODEL", "借款用途（枚举），申请时选择"),
    "title":              ("TEXT", "借款标题自由文本，与 purpose 高度重复"),
    "zip_code":           ("PLATFORM", "邮编前三位，高基数地理变量；与 addr_state 重叠，"
                                       "且地理变量做授信有公平信贷合规风险，本项目只留州"),
    "addr_state":         ("MODEL", "州，申请时填报"),

    # ---- 征信报告字段（申请时拉取，均为申请时点可得）----
    "dti":                ("MODEL", "负债收入比，申请时按征信负债计算"),
    "delinq_2yrs":        ("MODEL", "征信：近 2 年逾期 30+ 次数"),
    "earliest_cr_line":   ("MODEL", "征信：最早开户月份，用于算征信历史长度"),
    "fico_range_low":     ("MODEL", "征信：申请时 FICO 区间下界"),
    "fico_range_high":    ("MODEL", "征信：申请时 FICO 区间上界"),
    "inq_last_6mths":     ("MODEL", "征信：近 6 个月硬查询次数（多头借贷信号）"),
    "mths_since_last_delinq": ("MODEL", "征信：距上次逾期月数；缺失=从未逾期，单独成箱"),
    "mths_since_last_record": ("MODEL", "征信：距上次公共记录月数；缺失=无公共记录"),
    "open_acc":           ("MODEL", "征信：当前开立账户数"),
    "pub_rec":            ("MODEL", "征信：公共不良记录数"),
    "revol_bal":          ("MODEL", "征信：循环信贷余额"),
    "revol_util":         ("MODEL", "征信：循环额度使用率，零售风控最强单变量之一"),
    "total_acc":          ("MODEL", "征信：历史账户总数"),
    "initial_list_status": ("PLATFORM", "LC 挂牌方式（整笔/拆分），平台内部字段"),

    # ---- 放款后还款表现（全部泄漏）----
    "out_prncp":          ("LEAK", "剩余本金，放款后"),
    "out_prncp_inv":      ("LEAK", "投资人口径剩余本金，放款后"),
    "total_pymnt":        ("LEAK", "累计已还总额，放款后"),
    "total_pymnt_inv":    ("LEAK", "投资人口径累计已还，放款后"),
    "total_rec_prncp":    ("LEAK", "累计已还本金，放款后；用于反算经验 LGD"),
    "total_rec_int":      ("LEAK", "累计已还利息，放款后"),
    "total_rec_late_fee": ("LEAK", "累计滞纳金，放款后"),
    "recoveries":         ("LEAK", "核销后回收金额，放款后；用于反算经验 LGD"),
    "collection_recovery_fee": ("LEAK", "催收回收手续费，放款后"),
    "last_pymnt_d":       ("TARGET", "最后一次还款月份，用于推算违约发生的账龄 MOB"),
    "last_pymnt_amnt":    ("LEAK", "最后一次还款金额，放款后"),
    "next_pymnt_d":       ("LEAK", "下期应还月份，放款后"),
    "last_credit_pull_d": ("LEAK", "最近一次征信拉取月份，贷后监控动作"),
    "last_fico_range_high": ("LEAK", "最新 FICO 上界——最经典的泄漏字段，放款后重新拉的征信"),
    "last_fico_range_low":  ("LEAK", "最新 FICO 下界，同上"),

    "collections_12_mths_ex_med": ("MODEL", "征信：近 12 个月催收记录数（不含医疗），申请时可得"),
    "mths_since_last_major_derog": ("MODEL", "征信：距上次 90+ 逾期月数"),
    "policy_code":        ("PLATFORM", "LC 产品策略标记，本数据集近乎常数"),
    "application_type":   ("MODEL", "个人申请 / 联合申请，申请时确定"),

    # ---- 联合申请 / 第二申请人 ----
    "annual_inc_joint":   ("SPARSE", "联合申请专有，绝大多数样本为空"),
    "dti_joint":          ("SPARSE", "联合申请专有"),
    "verification_status_joint": ("SPARSE", "联合申请专有"),

    "acc_now_delinq":     ("MODEL", "征信：当前逾期账户数"),
    "tot_coll_amt":       ("MODEL", "征信：历史催收总金额"),
    "tot_cur_bal":        ("MODEL", "征信：全部账户当前余额合计"),
    "open_acc_6m":        ("MODEL", "征信：近 6 个月新开账户数（2015-12 起才全量采集，缺失率高）"),
    "open_act_il":        ("MODEL", "征信：当前活跃分期账户数（同上，晚期才采集）"),
    "open_il_12m":        ("MODEL", "征信：近 12 个月新开分期账户数"),
    "open_il_24m":        ("MODEL", "征信：近 24 个月新开分期账户数"),
    "mths_since_rcnt_il": ("MODEL", "征信：距最近一次开分期账户月数"),
    "total_bal_il":       ("MODEL", "征信：分期账户余额合计"),
    "il_util":            ("MODEL", "征信：分期账户额度使用率"),
    "open_rv_12m":        ("MODEL", "征信：近 12 个月新开循环账户数"),
    "open_rv_24m":        ("MODEL", "征信：近 24 个月新开循环账户数"),
    "max_bal_bc":         ("MODEL", "征信：单张信用卡最高余额"),
    "all_util":           ("MODEL", "征信：全账户额度使用率"),
    "total_rev_hi_lim":   ("MODEL", "征信：循环账户总授信额度"),
    "inq_fi":             ("MODEL", "征信：金融机构查询次数"),
    "total_cu_tl":        ("MODEL", "征信：信用社账户数"),
    "inq_last_12m":       ("MODEL", "征信：近 12 个月查询次数"),
    "acc_open_past_24mths": ("MODEL", "征信：近 24 个月新开账户数"),
    "avg_cur_bal":        ("MODEL", "征信：账户平均余额"),
    "bc_open_to_buy":     ("MODEL", "征信：信用卡可用额度"),
    "bc_util":            ("MODEL", "征信：信用卡额度使用率"),
    "chargeoff_within_12_mths": ("MODEL", "征信：近 12 个月核销笔数"),
    "delinq_amnt":        ("MODEL", "征信：当前逾期金额"),
    "mo_sin_old_il_acct": ("MODEL", "征信：最早分期账户账龄"),
    "mo_sin_old_rev_tl_op": ("MODEL", "征信：最早循环账户账龄"),
    "mo_sin_rcnt_rev_tl_op": ("MODEL", "征信：最近开立循环账户距今月数"),
    "mo_sin_rcnt_tl":     ("MODEL", "征信：最近开立任一账户距今月数"),
    "mort_acc":           ("MODEL", "征信：按揭账户数"),
    "mths_since_recent_bc": ("MODEL", "征信：距最近开立信用卡月数"),
    "mths_since_recent_bc_dlq": ("MODEL", "征信：距最近信用卡逾期月数"),
    "mths_since_recent_inq": ("MODEL", "征信：距最近查询月数"),
    "mths_since_recent_revol_delinq": ("MODEL", "征信：距最近循环账户逾期月数"),
    "num_accts_ever_120_pd": ("MODEL", "征信：历史曾 120+ 逾期账户数"),
    "num_actv_bc_tl":     ("MODEL", "征信：活跃信用卡数"),
    "num_actv_rev_tl":    ("MODEL", "征信：活跃循环账户数"),
    "num_bc_sats":        ("MODEL", "征信：状态正常的信用卡数"),
    "num_bc_tl":          ("MODEL", "征信：信用卡账户总数"),
    "num_il_tl":          ("MODEL", "征信：分期账户总数"),
    "num_op_rev_tl":      ("MODEL", "征信：开立中的循环账户数"),
    "num_rev_accts":      ("MODEL", "征信：循环账户总数"),
    "num_rev_tl_bal_gt_0": ("MODEL", "征信：有余额的循环账户数"),
    "num_sats":           ("MODEL", "征信：状态正常账户数"),
    "num_tl_120dpd_2m":   ("MODEL", "征信：近 2 个月 120+ 逾期账户数"),
    "num_tl_30dpd":       ("MODEL", "征信：当前 30+ 逾期账户数"),
    "num_tl_90g_dpd_24m": ("MODEL", "征信：近 24 个月 90+ 逾期账户数"),
    "num_tl_op_past_12m": ("MODEL", "征信：近 12 个月开立账户数"),
    "pct_tl_nvr_dlq":     ("MODEL", "征信：从未逾期账户占比"),
    "percent_bc_gt_75":   ("MODEL", "征信：使用率超 75% 的信用卡占比"),
    "pub_rec_bankruptcies": ("MODEL", "征信：破产记录数"),
    "tax_liens":          ("MODEL", "征信：税务留置权记录数"),
    "tot_hi_cred_lim":    ("MODEL", "征信：历史最高授信额度合计"),
    "total_bal_ex_mort":  ("MODEL", "征信：除按揭外余额合计"),
    "total_bc_limit":     ("MODEL", "征信：信用卡额度合计"),
    "total_il_high_credit_limit": ("MODEL", "征信：分期账户最高额度合计"),

    "revol_bal_joint":    ("SPARSE", "联合申请专有"),
    "sec_app_fico_range_low": ("SPARSE", "第二申请人专有"),
    "sec_app_fico_range_high": ("SPARSE", "第二申请人专有"),
    "sec_app_earliest_cr_line": ("SPARSE", "第二申请人专有"),
    "sec_app_inq_last_6mths": ("SPARSE", "第二申请人专有"),
    "sec_app_mort_acc":   ("SPARSE", "第二申请人专有"),
    "sec_app_open_acc":   ("SPARSE", "第二申请人专有"),
    "sec_app_revol_util": ("SPARSE", "第二申请人专有"),
    "sec_app_open_act_il": ("SPARSE", "第二申请人专有"),
    "sec_app_num_rev_accts": ("SPARSE", "第二申请人专有"),
    "sec_app_chargeoff_within_12_mths": ("SPARSE", "第二申请人专有"),
    "sec_app_collections_12_mths_ex_med": ("SPARSE", "第二申请人专有"),
    "sec_app_mths_since_last_major_derog": ("SPARSE", "第二申请人专有"),

    # ---- 放款后困难减免 / 债务和解（全部泄漏）----
    "hardship_flag":      ("LEAK", "是否申请过困难减免，放款后"),
    "hardship_type":      ("LEAK", "困难减免类型，放款后"),
    "hardship_reason":    ("LEAK", "困难减免原因，放款后"),
    "hardship_status":    ("LEAK", "困难减免状态，放款后"),
    "deferral_term":      ("LEAK", "延期期数，放款后"),
    "hardship_amount":    ("LEAK", "困难减免金额，放款后"),
    "hardship_start_date": ("LEAK", "困难减免起始日，放款后"),
    "hardship_end_date":  ("LEAK", "困难减免结束日，放款后"),
    "payment_plan_start_date": ("LEAK", "还款计划起始日，放款后"),
    "hardship_length":    ("LEAK", "困难减免时长，放款后"),
    "hardship_dpd":       ("LEAK", "困难减免时的逾期天数，放款后——直接泄漏标签"),
    "hardship_loan_status": ("LEAK", "困难减免时的贷款状态，放款后——直接泄漏标签"),
    "orig_projected_additional_accrued_interest": ("LEAK", "预计额外应计利息，放款后"),
    "hardship_payoff_balance_amount": ("LEAK", "困难减免时应付余额，放款后"),
    "hardship_last_payment_amount": ("LEAK", "困难减免期最后还款额，放款后"),
    "disbursement_method": ("PLATFORM", "放款方式（现金/直付），放款环节确定"),
    "debt_settlement_flag": ("LEAK", "是否进入债务和解，放款后——几乎等价于坏标签"),
    "debt_settlement_flag_date": ("LEAK", "债务和解标记日期，放款后"),
    "settlement_status":  ("LEAK", "债务和解状态，放款后"),
    "settlement_date":    ("LEAK", "债务和解日期，放款后"),
    "settlement_amount":  ("LEAK", "债务和解金额，放款后"),
    "settlement_percentage": ("LEAK", "债务和解比例，放款后"),
    "settlement_term":    ("LEAK", "债务和解期数，放款后"),
}


def _by(cat: str) -> list[str]:
    return [k for k, (c, _) in FIELDS.items() if c == cat]


MODEL_FEATURES = _by("MODEL")        # 申请时点可得，进候选池
PRICING_FEATURES = _by("PRICING")    # LC 定价输出，仅对照卡 / 收益测算用
LEAK_FEATURES = _by("LEAK")
TARGET_FIELDS = _by("TARGET")

# 建标签、算收益需要读进来但不进模型的列
AUX_FIELDS = ["id", "issue_d", "loan_status", "last_pymnt_d", "term",
              "int_rate", "grade", "sub_grade", "installment",
              "funded_amnt", "total_rec_prncp", "total_rec_int",
              "recoveries", "zip_code"]

# 全流程实际要从 1.67GB 原始 CSV 读进来的列
READ_COLUMNS = sorted(set(MODEL_FEATURES) | set(AUX_FIELDS))


def audit_table() -> pd.DataFrame:
    """逐列判定表，写进 docs/ 供面试时逐条解释。"""
    rows = [{"字段": k, "判定": c, "理由": why} for k, (c, why) in FIELDS.items()]
    df = pd.DataFrame(rows)
    order = {"MODEL": 0, "PRICING": 1, "TARGET": 2, "LEAK": 3,
             "SPARSE": 4, "TEXT": 5, "PLATFORM": 6, "ID": 7}
    df["_o"] = df["判定"].map(order)
    return df.sort_values(["_o", "字段"]).drop(columns="_o").reset_index(drop=True)


def check_coverage(columns) -> tuple[list[str], list[str]]:
    """校验：原始表里有没有我没判定过的列；判定表里有没有原始表不存在的列。
    任何一边非空都必须先补判定再跑，不能默认放行。"""
    cols = set(columns)
    known = set(FIELDS)
    return sorted(cols - known), sorted(known - cols)
