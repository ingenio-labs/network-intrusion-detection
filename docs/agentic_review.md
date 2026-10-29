# Agentic reviewer (design only)

An LLM agent (for example Claude on AWS Bedrock) that triages alerts **after** the model has flagged them. Nothing
here is built: this page is the design, and how I would prove that it helps.

## Why

The model keeps false alarms inside the budget, but analysts still open every incident and gather the same evidence
each time: what was unusual about the traffic, whether the source is known, whether it happened before. An agent can
gather that evidence and propose a verdict, so analysts spend their time deciding, not searching.

## Where it sits

- **Off the scoring path.** The model scores every flow in milliseconds; the agent never scores flows. It works
  asynchronously on the alert queue (diagram in [`production.md`](production.md)).
- **On incidents, not flows.** Flagged flows are first grouped into one incident per source. On this day the bad
  traffic came from about 800 source networks, so incidents number in the hundreds a day, while flagged flows number
  in the hundreds of thousands.

## What it does for each incident

It gathers evidence with **read-only tools**:

1. **The flows vs normal traffic:** each input compared with typical benign values for the same port (for example
   bytes out vs the benign median on that port).
2. **The model's view:** score, threshold, model version, and the inputs that pushed the score up the most.
3. **The rule baseline:** the port-lookup score, as a second opinion.
4. **Recent activity from the source:** its other flows and incidents in the last day.
5. **Threat intelligence:** whether the source is on a blocklist, and since when.
6. **Similar past incidents:** retrieval (RAG) over past incident notes and how they were resolved.

It returns a **structured verdict** with the same fields every time:

- **verdict:** likely attack, likely false alarm, or needs a human;
- **evidence:** the facts behind the verdict, each linked to the tool result it came from;
- **recommended action:** for example block the source, keep watching, or close as benign, for a person to approve;
- **summary:** two or three plain-English sentences for the analyst.

## What it can and cannot do

- **It can** shorten triage and push likely false alarms down the queue.
- **It cannot find attacks the model missed.** It only sees flagged traffic. Missed attacks are measured separately,
  with a reviewed random sample of unflagged traffic and with red-team test traffic ([`production.md`](production.md),
  section 5).
- **It cannot replace the model's score.** Per flow, an LLM is too slow and too costly, and its answers are not
  exactly repeatable.

## Guardrails

- **Humans approve any blocking.** The agent recommends; it never acts on the network.
- **It can lower an incident's priority, but never delete or close it.** Every incident stays in the queue.
- **Full audit log:** the inputs, every tool call and its result, the verdict, and the LLM and prompt versions.
- **Prompt injection:** everything it reads (flow fields, threat-intel text, past incident notes) is treated as
  data, never as instructions. Its tools are read-only, so a manipulated input cannot make it change anything.
- **Limits per incident:** a cap on tool calls, time (for example 60 seconds) and cost. When a limit is hit or a
  tool fails, the verdict is "needs a human".

## How to prove it helps

1. **Replay in shadow mode.** Run the agent on past alerts whose outcome analysts already know; nobody sees its
   verdicts. Measure:
   - **agreement with analysts**, overall and for each verdict;
   - **the share of real attacks it would call false alarms.** This must be near zero, and it is the go/no-go number;
   - **analyst time saved:** triage time with the agent's evidence and summary vs without.
2. **Small pilot.** One analyst team sees the verdicts as suggestions on a share of incidents. Compare triage time
   and decisions with incidents that had no suggestion, and ask analysts to rate the summaries.
3. **Then expand,** keeping the dismissed-attack share and the analyst ratings on the monitoring dashboard.
