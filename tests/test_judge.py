from verdict.judge import combine_orders, parse_json_reply, positional_to_score


def test_parse_plain_fenced_and_noisy():
    assert parse_json_reply('{"rationale": "ok", "score": 4}', "score")["score"] == 4
    assert parse_json_reply('```json\n{"score": 2, "rationale": "x"}\n```', "score")["score"] == 2
    noisy = 'Let me think {not json}. Final: {"rationale": "uses {braces} in text", "score": 5} done'
    assert parse_json_reply(noisy, "score")["score"] == 5
    assert parse_json_reply("score is 4", "score") is None
    assert parse_json_reply('{"winner": "B"}', "score") is None


def test_position_swap_logic():
    # x shown as A wins, x shown as B loses  => both orders favoured position A => flip
    s1, s2 = positional_to_score("A", True), positional_to_score("A", False)
    assert combine_orders(s1, s2) == (0.5, True)
    # consistent: x wins in both orders
    assert combine_orders(positional_to_score("A", True), positional_to_score("B", False)) == (1.0, False)
    assert combine_orders(None, None) == (None, False)
