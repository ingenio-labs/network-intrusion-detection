# Business framing

Who uses the model, what its errors cost, and how success is measured. Every number comes from this repo:
`notebooks/01_eda.ipynb`, `notebooks/02_model_development.ipynb` and `models/metrics.json`.

## Who uses the score, and what happens after an alert

- **User:** a security analyst who watches the network around the payment systems.
- **What the model gives:** a suspicion score from 0 to 1 for every network flow. Flows at or above the threshold
  become alerts, ranked by score.
- **After an alert:** flagged flows are grouped into one incident per source. Sources already on a threat-intel
  blocklist can be blocked automatically. Analysts triage the other incidents (confirm, then block or escalate), so
  they work on incidents, not on single flows.

## What each error costs

| Error | What happens | Cost for a payments company |
|---|---|---|
| **False alarm** (a benign flow is flagged) | An analyst spends time on it; at worst, a real merchant or service is blocked | Analyst hours, alert fatigue, lost transactions |
| **Missed attack** (a bad flow is not flagged) | The attack goes on unseen | A possible breach, fraud loss and PCI DSS exposure: logging, monitoring and intrusion detection around the cardholder data environment are PCI DSS requirements |

False alarms are frequent and cost time; missed attacks are rarer and can cost far more. So I cap false alarms at
what a team can handle and, inside that cap, catch as much bad traffic as possible.

## The target I chose: "not benign"

The data has three labels: benign 80.1%, malicious 16.0% and outlier 3.8%. The brief says "outlier or not", but this
is intrusion detection, and known attacks (malicious) must be caught too. With "outlier vs rest", malicious traffic
would count as normal, and the model would be punished for flagging real attacks. Malicious and outlier flows also
look the same: 8,728 groups of identical flows carry both labels. So the model predicts **not benign** (malicious or
outlier = 1), and recall is reported for each label. The API keeps the brief's field name, `is_outlier`.

## The success metric

**Recall inside a daily false-alarm budget, within a speed budget.**

- **Alert budget:** at most 0.01% of benign flows may be flagged. This network has about 1.57 million benign flows
  a day, so that is about **157 false alarms a day**, a volume a small team can review. A 1% budget would allow
  about 15,700.
- **Recall:** the share of bad flows caught inside that budget. Accuracy is not used, because "always benign" is
  already 80% accurate.
- **Speed budget:** one process must score at least **4,500 flows a second**, 10 times the busiest minute in the
  data (26,695 flows at 17:01 UTC, about 445 a second).

## Why speed matters here

A flow record arrives only after its connection ends, so scoring is near-real time, not inline: nothing waits for
the answer, the way a card payment waits for a fraud check. Speed still matters in two ways. **Keeping up with
bursts:** 63% of the day's flows arrived between 16:00 and 18:00 UTC. **Time to alert:** the sooner an alert
arrives, the sooner a source can be blocked. Scoring in small batches handles both: in batches of 1,000, the served
model scores about 75,000 flows a second on one CPU thread, while one flow at a time it would manage only about 110.

## Business impact on this day's traffic

| | Port rule (no machine learning) | Served model (gradient boosting) |
|---|---|---|
| Bad flows caught | 91.6% | **96.5%** |
| Bad flows missed a day | about 32,900 | **about 13,600** |
| False alarms a day | about 280 (1.8x the budget) | **about 40** (95% range 20 to 79) |

Compared with the simple port rule, the model catches about **19,000 more bad flows a day** and raises about **240
fewer false alarms a day** (7 times fewer). These are flows, not attacks: many flows belong to one attack, and they
are grouped into incidents.

**Limits of these numbers:** one day of one network (LUFlow, a UK university network, not a payments network), and
a benchmark with 20% bad traffic, far more than real traffic has. Before any claim in production, the budget and
the precision must be measured again on real volumes (`docs/production.md`).
