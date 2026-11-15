Judge vs human on 28 answers: exact agreement **89%**, binary (correct vs not) **93%**, Cohen's kappa **0.46**.

| human \ judge | correct | partially_correct | incorrect |
|---|---|---|---|
| correct | 24 | 1 | 0 |
| partially_correct | 1 | 0 | 0 |
| incorrect | 0 | 1 | 1 |

Disagreements:

- q13 (prompt v2): human correct, judge partially_correct – The assistant correctly states that setting reuse to true when creating the payment method allows it to be used for multiple reservations, which matches the reference. However, it omits the detail about including the reservationId to enable payment automations, making the answer incomplete.
- q19 (prompt v2): human partially_correct, judge correct – The assistant's answer correctly explains that a 403 Forbidden error means the user is authenticated but not permitted to make the request, including examples like wrong API key or plan restrictions, matching the reference. It also adds a helpful resolution tip without contradicting the reference. Human note: can also be permissions issue
- q27 (prompt v2): human incorrect, judge partially_correct – The assistant explains the existence of rate limits, the handling of HTTP 429 responses, and the headers indicating limits and remaining requests, but it does not specify the exact numeric limits (15 per second, 120 per minute, 5000 per hour) or that limits are shared across all account tokens as the reference does. Therefore, it is missing essential facts from the reference answer. Human note: the answer does not answer the question
