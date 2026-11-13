# 领域资料与供数约定

## 数据含义

cost 为不可拆分检修成本，budget 为总上限，repairable=false 的站不可选。failure_group 同组站可能一起不可用，但不假定概率独立。truth 只在 quality=valid 且非空时作为评价参照；模拟可用性应遮蔽被留出的真值，估计器不能再访问。protected_villages 是预算会关心的重点村，不表示可以忽略其他村。

## 现行摘要

`station_health/analysis.py` 生成现行资料摘要。源码和运行结果中的 scope 说明其统计边界，不能把未提供的推断当成已有能力。

## 供数接口

`GET /manifest` 返回 generation、tables（每表 count 与 sha256）及 metadata。`GET /tables/<name>?generation=<generation>` 返回同代 rows。供数换代时旧 generation 返回 HTTP 409，不允许将新代记录接入旧分析；不存在的表返回 404。表内容为稳定键唯一的完整同代记录，不是增量事件流。客户端对清单摘要进行核验。摘要使用 UTF-8、排序对象键、紧凑 JSON 的 SHA256，数组顺序保留供数端顺序。

字段与主键声明位于 `station_health/schema.json`，合成资料位于 `fixtures/dataset.json`。`metadata.cutoff` 是供数展示截点，不自动为分析选择训练截止点。服务可将整个 dataset 原子替换为新一代，但不在同一 generation 下更改内容。

## 结果保存

`reports.py` 保存现有摘要及来源表摘要，report_id 是内容摘要，不是随机编号。相同内容生成相同编号，临时文件写入完成后原子替换目标文件；历史内容是否保留由调用者选择不同目标路径。服务只监听本机，不接入生产系统。
