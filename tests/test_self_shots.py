"""The thinking rounds with same-shop examples, on a fake engine: no GPU, no gold answers."""
from types import SimpleNamespace

import duckdb

from datamate.pipeline import Pipeline, Shop

ANS = lambda sql: f"TYPE: number\n```sql\n{sql}\n```"


class FakeEngine:
    """Plain samples: question 0 always right (5), question 1 split 5 ways. Thinking samples answer 7 only when the chat
    shows the solved question of this shop."""
    def __init__(self):
        self.think_chats = []

    def params(self, **kw):
        return SimpleNamespace(**kw)

    def n_tokens(self, chat):
        return 10

    def chat(self, chats, params, *, thinking=False, lora=False, prefill=False):
        outs = []
        for n, c in enumerate(chats):
            user = c[1]["content"]
            if thinking:
                self.think_chats.append(user)
                texts = ["<think>\nok</think>\n" + ANS("SELECT 7" if "SOLVED QUESTIONS OF THIS SHOP" in user else "SELECT 100 + " + str(len(self.think_chats)))]
            elif "QUESTION: easy" in user:
                texts = [ANS("SELECT 5")] * params.n
            else:
                texts = [ANS(f"SELECT {20 + n * 10 + j}") for j in range(params.n)]
            outs.append(SimpleNamespace(outputs=[SimpleNamespace(text=t) for t in texts]))
        return outs


def run(self_shots, rounds=1):
    a = SimpleNamespace(k=8, temperature=0.7, context=49152, max_tokens=2048, seed=0, notes=False, repairs=0, think_k=4, think_below=0.75,
                        think_rounds=rounds, think_budget=1024, self_shots=self_shots, self_conf=0.875, rule_hints=0, threshold=0.5,
                        debt_route=False, lora=None, lora_thinking=False, fewshot_k=4, answer_unanswerable=True)
    p = Pipeline.__new__(Pipeline)
    p.a, p.engine, p.half, p.systems, p.bank, p.rules = a, FakeEngine(), 4, ["S0", "S1"], None, []
    shop = Shop("s", duckdb.connect(), {}, {}, ctx="CTX")
    todo = [{"id": "a", "question": "easy how many orders in March", "as_of_date": "2026-01-01", "_hits": [], "_periods": [], "_oor": False},
            {"id": "b", "question": "hard how many orders in April", "as_of_date": "2026-01-01", "_hits": [], "_periods": [], "_oor": False}]
    chats = [[[{"role": "system", "content": s}, {"role": "user", "content": "CTX\n\nQUESTION: " + q["question"]}] for s in p.systems] for q in todo]
    texts, preds, errs, pools, _, _, _ = p.sample(shop, todo, chats, lambda: False)
    return p, [d[0] for d in p.decide(todo, preds, pools)], [len(t) for t in texts]


def test_plain_thinking_cannot_settle_the_hard_question():
    p, final, n = run(self_shots=0)
    assert final[0] == {"type": "number", "answer": 5} and final[1]["type"] == "abstain"
    assert n == [8, 12] and not any("SOLVED QUESTIONS" in c for c in p.engine.think_chats)


def test_solved_questions_of_the_shop_reach_the_hard_one_and_a_second_round_adds_samples():
    p, final, n = run(self_shots=2, rounds=2)
    assert all("SELECT 5" in c and "easy how many" in c for c in p.engine.think_chats)  # the easy question's SQL is the example
    assert n == [8, 16]                                    # 4/12 after round one is below --think-below: a second round
    assert final[1] == {"type": "number", "answer": 7}     # 8 of 16 samples agree
