# Full transcript: exact prompts and exact raw LLM responses, all 20 questions

45 calls total, all cache hits (no new cost).

## 1. F118 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how justifiable people consider homosexuality, on a scale from 1 (never justifiable) to 10 (always justifiable). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "always justifiable" (attitudes became more accepting).
Direction B: average answer moved toward "never justifiable" (attitudes became less accepting).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines discussing legalization or expansion of same-sex marriage rights in the US.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["same-sex marriage", "gay marriage", "marriage equality"],
      "nyt_sections": ["U.S.", "Politics"]
    },
    {
      "indicator": "More GDELT events reporting government or legislative bodies passing pro-LGBTQ+ laws or policies.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["05", "08"],
      "gdelt_actor_types": ["GOV", "LEG"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in NYT opinion pieces or editorials supporting LGBTQ+ rights and acceptance.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["LGBTQ rights", "gay rights", "homosexuality acceptance"],
      "nyt_sections": ["Opinion"]
    },
    {
      "indicator": "More GDELT protest events by LGBTQ+ advocacy groups or allies demanding equal rights and social acceptance.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in NYT coverage of positive media representation of LGBTQ+ individuals and culture.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["LGBTQ representation", "gay pride", "coming out stories"],
      "nyt_sections": ["Arts", "Culture"]
    }
  ],
  "B": [
    {
      "indicator": "Increase in New York Times headlines reporting legislative attempts to restrict or roll back LGBTQ+ rights.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["ban same-sex marriage", "anti-gay legislation", "LGBTQ restrictions"],
      "nyt_sections": ["U.S.", "Politics"]
    },
    {
      "indicator": "More GDELT events involving religious groups or conservative organizations publicly opposing homosexuality or LGBTQ+ rights.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01", "11"],
      "gdelt_actor_types": ["REL"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in NYT opinion pieces or editorials expressing skepticism or opposition to homosexuality acceptance.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["traditional marriage", "family values", "homosexuality criticism"],
      "nyt_sections": ["Opinion"]
    },
    {
      "indicator": "More GDELT protest events by groups opposing LGBTQ+ rights or promoting conservative social values.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["REL", "CVL"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in NYT coverage of controversies or conflicts related to LGBTQ+ issues, highlighting social backlash.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["LGBTQ controversy", "gay rights backlash", "homosexuality debate"],
      "nyt_sections": ["U.S.", "Politics"]
    }
  ]
}
```

## 2. F118 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how justifiable people consider homosexuality, on a scale from 1 (never justifiable) to 10 (always justifiable). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "always justifiable" (attitudes became more accepting).
Direction B: average answer moved toward "never justifiable" (attitudes became less accepting).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines discussing legalization or expansion of same-sex marriage rights in the US." (source: nyt, predicted: increase)
   Measured: 74/10742 matching articles (6.89 per 1,000) in 2006 vs. 52/9840 (5.28 per 1,000) in 2011 [sections: U.S., Politics]
2. "More GDELT events reporting government or legislative bodies passing pro-LGBTQ+ laws or policies." (source: gdelt, predicted: increase)
   Measured: 24754/969651 matching events (25.53 per 1,000) in 2006 vs. 251925/10259907 (24.55 per 1,000) in 2011 [actors: GOV, LEG]; avg tone 6.46 vs. 6.53
3. "Increase in NYT opinion pieces or editorials supporting LGBTQ+ rights and acceptance." (source: nyt, predicted: increase)
   Measured: 1/16492 matching articles (0.06 per 1,000) in 2006 vs. 7/9196 (0.76 per 1,000) in 2011 [sections: Opinion]
4. "More GDELT protest events by LGBTQ+ advocacy groups or allies demanding equal rights and social acceptance." (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL]; avg tone 5.51 vs. 6.09
5. "Increase in NYT coverage of positive media representation of LGBTQ+ individuals and culture." (source: nyt, predicted: increase)
   Measured: 8/13630 matching articles (0.59 per 1,000) in 2006 vs. 0/10269 (0.00 per 1,000) in 2011 [sections: Arts, Culture]

=== Direction B ===
1. "Increase in New York Times headlines reporting legislative attempts to restrict or roll back LGBTQ+ rights." (source: nyt, predicted: increase)
   Measured: 12/10742 matching articles (1.12 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Politics]
2. "More GDELT events involving religious groups or conservative organizations publicly opposing homosexuality or LGBTQ+ rights." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]
3. "Increase in NYT opinion pieces or editorials expressing skepticism or opposition to homosexuality acceptance." (source: nyt, predicted: increase)
   Measured: 2/16492 matching articles (0.12 per 1,000) in 2006 vs. 3/9196 (0.33 per 1,000) in 2011 [sections: Opinion]
4. "More GDELT protest events by groups opposing LGBTQ+ rights or promoting conservative social values." (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: REL, CVL]; avg tone 5.51 vs. 6.09
5. "Increase in NYT coverage of controversies or conflicts related to LGBTQ+ issues, highlighting social backlash." (source: nyt, predicted: increase)
   Measured: 0/10742 matching articles (0.00 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Politics]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "A", "reasoning": "The evidence for direction A shows some increase in supportive opinion pieces and protests by LGBTQ+ advocacy groups, while direction B indicators mostly show no increase or very low levels of opposition events and articles. The lack of increase in opposition-related indicators and some positive movement in supportive indicators favors direction A.", "requested_evidence": []}
```

## 3. A106 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people are a member of, and how active they are in, a voluntary organization not covered by other specific categories (such as clubs or civic groups), on a scale from 0 (not a member) to 2 (active member). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "active member" (civic participation increased).
Direction B: average answer moved toward "not a member" (civic participation decreased).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times articles highlighting growth or success stories of local clubs, civic groups, and voluntary organizations.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["voluntary organization", "civic group", "community club", "active membership", "volunteering"],
      "nyt_sections": ["U.S.", "Society", "Opinion"]
    },
    {
      "indicator": "More GDELT events coded as '03 EXPRESS INTENT TO COOPERATE' involving civilians or community groups in the United States.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["03"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_location_contains": ["United States"],
      "gdelt_goldstein": "more_positive"
    },
    {
      "indicator": "Increase in New York Times coverage of volunteer recruitment drives, community service initiatives, and civic engagement campaigns.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["volunteer recruitment", "community service", "civic engagement", "join club", "active member"],
      "nyt_sections": ["U.S.", "Society"]
    },
    {
      "indicator": "More GDELT events coded as '06 ENGAGE IN MATERIAL COOPERATION' involving civilian actors or NGOs in the U.S., indicating active collaboration.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["06"],
      "gdelt_actor_types": ["CVL", "REL", "EDU"],
      "gdelt_location_contains": ["United States"],
      "gdelt_goldstein": "more_positive"
    },
    {
      "indicator": "Increase in New York Times opinion pieces or editorials encouraging civic participation and membership in voluntary organizations.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["civic participation", "voluntary organization", "community involvement", "active membership"],
      "nyt_sections": ["Opinion"]
    }
  ],
  "B": [
    {
      "indicator": "Increase in New York Times articles reporting decline or challenges faced by voluntary organizations and civic groups.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["decline in membership", "voluntary organization struggles", "civic group challenges", "drop in volunteers"],
      "nyt_sections": ["U.S.", "Society"]
    },
    {
      "indicator": "More GDELT events coded as '14 PROTEST' or '11 DISAPPROVE' involving civilians expressing dissatisfaction with community organizations or civic institutions.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["11", "14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_location_contains": ["United States"],
      "gdelt_tone": "more_negative",
      "gdelt_goldstein": "more_negative"
    },
    {
      "indicator": "Increase in New York Times coverage of people quitting or withdrawing from clubs, civic groups, or voluntary organizations.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["quit club", "leave civic group", "drop membership", "inactive member"],
      "nyt_sections": ["U.S.", "Society"]
    },
    {
      "indicator": "More GDELT events coded as '12 REJECT' involving civilian actors rejecting participation or membership in voluntary organizations.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["12"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_location_contains": ["United States"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in New York Times opinion pieces or editorials discussing civic disengagement, apathy, or decline in community involvement.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["civic disengagement", "community apathy", "decline in volunteering", "inactive membership"],
      "nyt_sections": ["Opinion"]
    }
  ]
}
```

## 4. A106 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people are a member of, and how active they are in, a voluntary organization not covered by other specific categories (such as clubs or civic groups), on a scale from 0 (not a member) to 2 (active member). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "active member" (civic participation increased).
Direction B: average answer moved toward "not a member" (civic participation decreased).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times articles highlighting growth or success stories of local clubs, civic groups, and voluntary organizations." (source: nyt, predicted: increase)
   Measured: 11/28294 matching articles (0.39 per 1,000) in 2006 vs. 3/20383 (0.15 per 1,000) in 2011 [sections: U.S., Society, Opinion]
2. "More GDELT events coded as '03 EXPRESS INTENT TO COOPERATE' involving civilians or community groups in the United States." (source: gdelt, predicted: increase)
   Measured: 2322/969651 matching events (2.39 per 1,000) in 2006 vs. 56062/10259907 (5.46 per 1,000) in 2011 [actors: CVL; location contains: United States]; avg Goldstein 4.29 vs. 4.19
3. "Increase in New York Times coverage of volunteer recruitment drives, community service initiatives, and civic engagement campaigns." (source: nyt, predicted: increase)
   Measured: 3/11802 matching articles (0.25 per 1,000) in 2006 vs. 4/11187 (0.36 per 1,000) in 2011 [sections: U.S., Society]
4. "More GDELT events coded as '06 ENGAGE IN MATERIAL COOPERATION' involving civilian actors or NGOs in the U.S., indicating active collaboration." (source: gdelt, predicted: increase)
   Measured: 1265/969651 matching events (1.30 per 1,000) in 2006 vs. 25139/10259907 (2.45 per 1,000) in 2011 [actors: CVL, REL, EDU; location contains: United States]; avg Goldstein 6.12 vs. 6.11
5. "Increase in New York Times opinion pieces or editorials encouraging civic participation and membership in voluntary organizations." (source: nyt, predicted: increase)
   Measured: 0/16492 matching articles (0.00 per 1,000) in 2006 vs. 0/9196 (0.00 per 1,000) in 2011 [sections: Opinion]

=== Direction B ===
1. "Increase in New York Times articles reporting decline or challenges faced by voluntary organizations and civic groups." (source: nyt, predicted: increase)
   Measured: 0/11802 matching articles (0.00 per 1,000) in 2006 vs. 0/11187 (0.00 per 1,000) in 2011 [sections: U.S., Society]
2. "More GDELT events coded as '14 PROTEST' or '11 DISAPPROVE' involving civilians expressing dissatisfaction with community organizations or civic institutions." (source: gdelt, predicted: increase)
   Measured: 2933/969651 matching events (3.02 per 1,000) in 2006 vs. 34879/10259907 (3.40 per 1,000) in 2011 [actors: CVL; location contains: United States]; avg tone 5.29 vs. 5.41; avg Goldstein -2.43 vs. -2.49
3. "Increase in New York Times coverage of people quitting or withdrawing from clubs, civic groups, or voluntary organizations." (source: nyt, predicted: increase)
   Measured: 0/11802 matching articles (0.00 per 1,000) in 2006 vs. 0/11187 (0.00 per 1,000) in 2011 [sections: U.S., Society]
4. "More GDELT events coded as '12 REJECT' involving civilian actors rejecting participation or membership in voluntary organizations." (source: gdelt, predicted: increase)
   Measured: 1169/969651 matching events (1.21 per 1,000) in 2006 vs. 17881/10259907 (1.74 per 1,000) in 2011 [actors: CVL; location contains: United States]; avg tone 5.71 vs. 5.56
5. "Increase in New York Times opinion pieces or editorials discussing civic disengagement, apathy, or decline in community involvement." (source: nyt, predicted: increase)
   Measured: 0/16492 matching articles (0.00 per 1,000) in 2006 vs. 0/9196 (0.00 per 1,000) in 2011 [sections: Opinion]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "A", "reasoning": "The evidence strongly favors Direction A, with substantial increases in GDELT events indicating cooperation and material cooperation involving civilians and community groups, as well as a modest increase in NYT coverage of volunteer recruitment and civic engagement campaigns. In contrast, indicators for Direction B show no increase in NYT articles about decline or quitting, and only a small increase in protest and rejection events that are less pronounced than the cooperation events. Overall, the data suggest increased civic participation.", "requested_evidence": []}
```

## 5. E222 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people have personally taken a form of political action not covered by other specific categories, on a scale from 1 (have done this) to 3 (would never do this). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "have done this" (people became more willing to take this kind of political action).
Direction B: average answer moved toward "would never do this" (people became less willing to take this kind of political action).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines or abstracts mentioning grassroots political activism or unconventional political actions by individuals.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["grassroots activism", "political protest", "civil disobedience", "unconventional political action", "political demonstration"],
      "nyt_sections": ["U.S.", "Politics"]
    },
    {
      "indicator": "More GDELT events coded as '14 PROTEST' involving civilians or general public actors in the United States.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_location_contains": ["United States"]
    },
    {
      "indicator": "Increase in New York Times coverage of political rallies, sit-ins, or other forms of direct political action by citizens.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["political rally", "sit-in", "direct action", "civil disobedience", "activist demonstration"],
      "nyt_sections": ["U.S.", "Politics"]
    },
    {
      "indicator": "More GDELT events with root code '10 DEMAND' or '11 DISAPPROVE' involving civilians or activist groups in the U.S., indicating increased political demands or expressions of dissatisfaction.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["10", "11"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_location_contains": ["United States"]
    },
    {
      "indicator": "Increase in New York Times opinion pieces or editorials encouraging or discussing unconventional political participation or activism.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["political participation", "activism", "civic engagement", "unconventional political action"],
      "nyt_sections": ["Opinion"]
    }
  ],
  "B": [
    {
      "indicator": "Decrease in New York Times headlines or abstracts mentioning grassroots political activism or unconventional political actions by individuals.",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["grassroots activism", "political protest", "civil disobedience", "unconventional political action", "political demonstration"],
      "nyt_sections": ["U.S.", "Politics"]
    },
    {
      "indicator": "Fewer GDELT events coded as '14 PROTEST' involving civilians or general public actors in the United States.",
      "source": "gdelt",
      "expected_change": "decrease",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_location_contains": ["United States"]
    },
    {
      "indicator": "Decrease in New York Times coverage of political rallies, sit-ins, or other forms of direct political action by citizens.",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["political rally", "sit-in", "direct action", "civil disobedience", "activist demonstration"],
      "nyt_sections": ["U.S.", "Politics"]
    },
    {
      "indicator": "Fewer GDELT events with root code '10 DEMAND' or '11 DISAPPROVE' involving civilians or activist groups in the U.S., indicating reduced political demands or expressions of dissatisfaction.",
      "source": "gdelt",
      "expected_change": "decrease",
      "gdelt_event_codes": ["10", "11"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_location_contains": ["United States"]
    },
    {
      "indicator": "Decrease in New York Times opinion pieces or editorials encouraging or discussing unconventional political participation or activism.",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["political participation", "activism", "civic engagement", "unconventional political action"],
      "nyt_sections": ["Opinion"]
    }
  ]
}
```

## 6. E222 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people have personally taken a form of political action not covered by other specific categories, on a scale from 1 (have done this) to 3 (would never do this). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "have done this" (people became more willing to take this kind of political action).
Direction B: average answer moved toward "would never do this" (people became less willing to take this kind of political action).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines or abstracts mentioning grassroots political activism or unconventional political actions by individuals." (source: nyt, predicted: increase)
   Measured: 4/10742 matching articles (0.37 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Politics]
2. "More GDELT events coded as '14 PROTEST' involving civilians or general public actors in the United States." (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL; location contains: United States]
3. "Increase in New York Times coverage of political rallies, sit-ins, or other forms of direct political action by citizens." (source: nyt, predicted: increase)
   Measured: 10/10742 matching articles (0.93 per 1,000) in 2006 vs. 2/9840 (0.20 per 1,000) in 2011 [sections: U.S., Politics]
4. "More GDELT events with root code '10 DEMAND' or '11 DISAPPROVE' involving civilians or activist groups in the U.S., indicating increased political demands or expressions of dissatisfaction." (source: gdelt, predicted: increase)
   Measured: 3222/969651 matching events (3.32 per 1,000) in 2006 vs. 39518/10259907 (3.85 per 1,000) in 2011 [actors: CVL; location contains: United States]
5. "Increase in New York Times opinion pieces or editorials encouraging or discussing unconventional political participation or activism." (source: nyt, predicted: increase)
   Measured: 7/16492 matching articles (0.42 per 1,000) in 2006 vs. 10/9196 (1.09 per 1,000) in 2011 [sections: Opinion]

=== Direction B ===
1. "Decrease in New York Times headlines or abstracts mentioning grassroots political activism or unconventional political actions by individuals." (source: nyt, predicted: decrease)
   Measured: 4/10742 matching articles (0.37 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Politics]
2. "Fewer GDELT events coded as '14 PROTEST' involving civilians or general public actors in the United States." (source: gdelt, predicted: decrease)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL; location contains: United States]
3. "Decrease in New York Times coverage of political rallies, sit-ins, or other forms of direct political action by citizens." (source: nyt, predicted: decrease)
   Measured: 10/10742 matching articles (0.93 per 1,000) in 2006 vs. 2/9840 (0.20 per 1,000) in 2011 [sections: U.S., Politics]
4. "Fewer GDELT events with root code '10 DEMAND' or '11 DISAPPROVE' involving civilians or activist groups in the U.S., indicating reduced political demands or expressions of dissatisfaction." (source: gdelt, predicted: decrease)
   Measured: 3222/969651 matching events (3.32 per 1,000) in 2006 vs. 39518/10259907 (3.85 per 1,000) in 2011 [actors: CVL; location contains: United States]
5. "Decrease in New York Times opinion pieces or editorials encouraging or discussing unconventional political participation or activism." (source: nyt, predicted: decrease)
   Measured: 7/16492 matching articles (0.42 per 1,000) in 2006 vs. 10/9196 (1.09 per 1,000) in 2011 [sections: Opinion]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "A", "reasoning": "Most indicators strongly support Direction A, showing increases in protest events, political demands, and opinion pieces encouraging activism, despite a decrease in some New York Times coverage of rallies. The large increase in GDELT-coded protest and demand events involving civilians is particularly compelling evidence of increased willingness to take political action.", "requested_evidence": []}
```

## 7. E035 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people think incomes should be made more equal or that larger income differences are needed as incentives, on a scale from 1 (incomes should be made more equal) to 10 (we need larger income differences as incentives). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "we need larger income differences as incentives" (people became more accepting of income inequality).
Direction B: average answer moved toward "incomes should be made more equal" (people became more in favor of income equality).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines emphasizing the importance of incentives and rewards for economic growth and innovation.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["income incentives", "economic growth", "reward innovation", "income differences", "market incentives"],
      "nyt_sections": ["Business", "Economy", "Opinion"]
    },
    {
      "indicator": "More GDELT events coded as '01 MAKE PUBLIC STATEMENT' by business actors promoting free market policies and defending income inequality as necessary.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["BUS"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in NYT opinion pieces or editorials arguing that larger income differences are needed to motivate productivity and entrepreneurship.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["income inequality justified", "need for incentives", "entrepreneurship rewards", "economic motivation"],
      "nyt_sections": ["Opinion"]
    },
    {
      "indicator": "More GDELT '10 DEMAND' or '11 DISAPPROVE' events involving government or business actors opposing policies aimed at redistributing income or raising taxes on the wealthy.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["10", "11"],
      "gdelt_actor_types": ["GOV", "BUS"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in NYT coverage highlighting success stories of wealthy entrepreneurs and the benefits of income inequality for innovation and job creation.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["wealth creation", "entrepreneur success", "income inequality benefits", "job creation"],
      "nyt_sections": ["Business", "U.S."]
    }
  ],
  "B": [
    {
      "indicator": "Increase in New York Times headlines and articles advocating for income equality, social justice, and reducing the wealth gap.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["income equality", "reduce wealth gap", "social justice", "economic fairness", "redistribution"],
      "nyt_sections": ["U.S.", "Opinion", "Business"]
    },
    {
      "indicator": "More GDELT '14 PROTEST' events involving civilians and labor unions demanding higher minimum wages, progressive taxation, and income redistribution policies.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in NYT opinion pieces criticizing income inequality and calling for stronger government intervention to promote economic equality.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["income inequality criticism", "government intervention", "economic equality", "tax the rich"],
      "nyt_sections": ["Opinion"]
    },
    {
      "indicator": "More GDELT '08 YIELD' or '12 REJECT' events where government actors accept or propose policies aimed at reducing income inequality, such as raising taxes on the wealthy or expanding social programs.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["08", "12"],
      "gdelt_actor_types": ["GOV"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in NYT coverage of social movements and political campaigns focused on economic justice, minimum wage increases, and closing the income gap.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["economic justice movement", "minimum wage increase", "closing income gap", "fair wages"],
      "nyt_sections": ["U.S.", "Politics"]
    }
  ]
}
```

## 8. E035 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people think incomes should be made more equal or that larger income differences are needed as incentives, on a scale from 1 (incomes should be made more equal) to 10 (we need larger income differences as incentives). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "we need larger income differences as incentives" (people became more accepting of income inequality).
Direction B: average answer moved toward "incomes should be made more equal" (people became more in favor of income equality).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines emphasizing the importance of incentives and rewards for economic growth and innovation." (source: nyt, predicted: increase)
   Measured: 42/31771 matching articles (1.32 per 1,000) in 2006 vs. 30/15414 (1.95 per 1,000) in 2011 [sections: Business, Economy, Opinion]
2. "More GDELT events coded as '01 MAKE PUBLIC STATEMENT' by business actors promoting free market policies and defending income inequality as necessary." (source: gdelt, predicted: increase)
   Measured: 14788/969651 matching events (15.25 per 1,000) in 2006 vs. 114709/10259907 (11.18 per 1,000) in 2011 [actors: BUS]; avg tone 7.23 vs. 5.98
3. "Increase in NYT opinion pieces or editorials arguing that larger income differences are needed to motivate productivity and entrepreneurship." (source: nyt, predicted: increase)
   Measured: 0/16492 matching articles (0.00 per 1,000) in 2006 vs. 0/9196 (0.00 per 1,000) in 2011 [sections: Opinion]
4. "More GDELT '10 DEMAND' or '11 DISAPPROVE' events involving government or business actors opposing policies aimed at redistributing income or raising taxes on the wealthy." (source: gdelt, predicted: increase)
   Measured: 22985/969651 matching events (23.70 per 1,000) in 2006 vs. 181342/10259907 (17.67 per 1,000) in 2011 [actors: GOV, BUS]; avg tone 5.75 vs. 5.57
5. "Increase in NYT coverage highlighting success stories of wealthy entrepreneurs and the benefits of income inequality for innovation and job creation." (source: nyt, predicted: increase)
   Measured: 4/26020 matching articles (0.15 per 1,000) in 2006 vs. 21/16043 (1.31 per 1,000) in 2011 [sections: Business, U.S.]

=== Direction B ===
1. "Increase in New York Times headlines and articles advocating for income equality, social justice, and reducing the wealth gap." (source: nyt, predicted: increase)
   Measured: 6/42512 matching articles (0.14 per 1,000) in 2006 vs. 4/25239 (0.16 per 1,000) in 2011 [sections: U.S., Opinion, Business]
2. "More GDELT '14 PROTEST' events involving civilians and labor unions demanding higher minimum wages, progressive taxation, and income redistribution policies." (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL]; avg tone 5.51 vs. 6.09
3. "Increase in NYT opinion pieces criticizing income inequality and calling for stronger government intervention to promote economic equality." (source: nyt, predicted: increase)
   Measured: 1/16492 matching articles (0.06 per 1,000) in 2006 vs. 1/9196 (0.11 per 1,000) in 2011 [sections: Opinion]
4. "More GDELT '08 YIELD' or '12 REJECT' events where government actors accept or propose policies aimed at reducing income inequality, such as raising taxes on the wealthy or expanding social programs." (source: gdelt, predicted: increase)
   Measured: 11929/969651 matching events (12.30 per 1,000) in 2006 vs. 111154/10259907 (10.83 per 1,000) in 2011 [actors: GOV]; avg tone 5.80 vs. 5.73
5. "Increase in NYT coverage of social movements and political campaigns focused on economic justice, minimum wage increases, and closing the income gap." (source: nyt, predicted: increase)
   Measured: 4/10742 matching articles (0.37 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Politics]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "A", "reasoning": "Indicators for Direction A show a clearer increase in acceptance of income inequality, such as a rise in NYT headlines emphasizing incentives (1.32 to 1.95 per 1,000), increased coverage of wealthy entrepreneurs' success stories (0.15 to 1.31 per 1,000), despite some decreases in GDELT events by business and government actors opposing redistribution. Direction B indicators show only slight or ambiguous increases in advocacy for income equality, with some measures even decreasing, such as NYT coverage of social movements (0.37 to 0.00 per 1,000). Overall, the evidence more strongly supports Direction A.", "requested_evidence": []}
```

## 9. E229 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people see civil rights that protect individual liberty against oppression as an essential characteristic of democracy, on a scale from 0 (against democracy) to 10 (an essential characteristic of democracy). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "an essential characteristic of democracy" (people placed more importance on civil-rights protections as part of democracy).
Direction B: average answer moved toward lower importance (people placed less importance on civil-rights protections as part of democracy).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines emphasizing the importance of civil rights protections as fundamental to democracy.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["civil rights", "essential to democracy", "individual liberty", "democracy protection"],
      "nyt_sections": ["U.S.", "Opinion"]
    },
    {
      "indicator": "More GDELT events coded as public statements (01) by government or judiciary actors affirming civil rights as core democratic values.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["GOV", "JUD"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in protests (14) recorded in GDELT advocating for civil rights protections and opposing oppression.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "More New York Times articles discussing legal rulings or legislative actions strengthening civil rights protections.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["civil rights legislation", "court ruling", "civil liberties", "democracy"],
      "nyt_sections": ["U.S.", "Politics"]
    },
    {
      "indicator": "Increase in GDELT events involving demands (10) or appeals (02) by civil society groups or media for stronger civil rights enforcement.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["10", "02"],
      "gdelt_actor_types": ["MED", "CVL"]
    }
  ],
  "B": [
    {
      "indicator": "Decrease in New York Times headlines emphasizing civil rights as essential to democracy, with more focus on security or order.",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["civil rights", "essential to democracy", "individual liberty"],
      "nyt_sections": ["U.S.", "Opinion"]
    },
    {
      "indicator": "Fewer GDELT public statements (01) by government or judiciary actors affirming civil rights protections as core democratic values.",
      "source": "gdelt",
      "expected_change": "decrease",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["GOV", "JUD"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Decrease in protests (14) advocating for civil rights protections recorded in GDELT, possibly replaced by protests focusing on security or other issues.",
      "source": "gdelt",
      "expected_change": "decrease",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"]
    },
    {
      "indicator": "Fewer New York Times articles covering legal or legislative advances in civil rights protections; possibly more coverage on restrictions or security measures.",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["civil rights legislation", "court ruling", "civil liberties"],
      "nyt_sections": ["U.S.", "Politics"]
    },
    {
      "indicator": "Decrease in GDELT events involving demands (10) or appeals (02) by civil society or media for stronger civil rights enforcement.",
      "source": "gdelt",
      "expected_change": "decrease",
      "gdelt_event_codes": ["10", "02"],
      "gdelt_actor_types": ["MED", "CVL"]
    }
  ]
}
```

## 10. E229 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people see civil rights that protect individual liberty against oppression as an essential characteristic of democracy, on a scale from 0 (against democracy) to 10 (an essential characteristic of democracy). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "an essential characteristic of democracy" (people placed more importance on civil-rights protections as part of democracy).
Direction B: average answer moved toward lower importance (people placed less importance on civil-rights protections as part of democracy).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines emphasizing the importance of civil rights protections as fundamental to democracy." (source: nyt, predicted: increase)
   Measured: 80/27234 matching articles (2.94 per 1,000) in 2006 vs. 40/19036 (2.10 per 1,000) in 2011 [sections: U.S., Opinion]
2. "More GDELT events coded as public statements (01) by government or judiciary actors affirming civil rights as core democratic values." (source: gdelt, predicted: increase)
   Measured: 42737/969651 matching events (44.07 per 1,000) in 2006 vs. 380898/10259907 (37.12 per 1,000) in 2011 [actors: GOV, JUD]; avg tone 5.43 vs. 5.18
3. "Increase in protests (14) recorded in GDELT advocating for civil rights protections and opposing oppression." (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL]; avg tone 5.51 vs. 6.09
4. "More New York Times articles discussing legal rulings or legislative actions strengthening civil rights protections." (source: nyt, predicted: increase)
   Measured: 120/10742 matching articles (11.17 per 1,000) in 2006 vs. 25/9840 (2.54 per 1,000) in 2011 [sections: U.S., Politics]
5. "Increase in GDELT events involving demands (10) or appeals (02) by civil society groups or media for stronger civil rights enforcement." (source: gdelt, predicted: increase)
   Measured: 11702/969651 matching events (12.07 per 1,000) in 2006 vs. 118441/10259907 (11.54 per 1,000) in 2011 [actors: MED, CVL]

=== Direction B ===
1. "Decrease in New York Times headlines emphasizing civil rights as essential to democracy, with more focus on security or order." (source: nyt, predicted: decrease)
   Measured: 80/27234 matching articles (2.94 per 1,000) in 2006 vs. 40/19036 (2.10 per 1,000) in 2011 [sections: U.S., Opinion]
2. "Fewer GDELT public statements (01) by government or judiciary actors affirming civil rights protections as core democratic values." (source: gdelt, predicted: decrease)
   Measured: 42737/969651 matching events (44.07 per 1,000) in 2006 vs. 380898/10259907 (37.12 per 1,000) in 2011 [actors: GOV, JUD]; avg tone 5.43 vs. 5.18
3. "Decrease in protests (14) advocating for civil rights protections recorded in GDELT, possibly replaced by protests focusing on security or other issues." (source: gdelt, predicted: decrease)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL]
4. "Fewer New York Times articles covering legal or legislative advances in civil rights protections; possibly more coverage on restrictions or security measures." (source: nyt, predicted: decrease)
   Measured: 78/10742 matching articles (7.26 per 1,000) in 2006 vs. 18/9840 (1.83 per 1,000) in 2011 [sections: U.S., Politics]
5. "Decrease in GDELT events involving demands (10) or appeals (02) by civil society or media for stronger civil rights enforcement." (source: gdelt, predicted: decrease)
   Measured: 11702/969651 matching events (12.07 per 1,000) in 2006 vs. 118441/10259907 (11.54 per 1,000) in 2011 [actors: MED, CVL]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "B", "reasoning": "All indicators show a decrease in emphasis on civil rights protections as essential to democracy between 2006 and 2011, including fewer New York Times headlines and articles, fewer government and judiciary affirmations, and a slight decrease in demands for stronger civil rights enforcement. Although protests advocating civil rights increased slightly, this is outweighed by the consistent decreases in other indicators.", "requested_evidence": []}
```

## 11. F063 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how important God is in people's lives, on a scale from 1 (not at all important) to 10 (very important). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "very important" (religion became more central to people's lives).
Direction B: average answer moved toward "not at all important" (people became more secular, religion less central).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines or abstracts discussing the rise or strengthening of religious faith or God in American life.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["rise of faith", "God important", "religion central", "spiritual revival", "religious resurgence"],
      "nyt_sections": ["U.S.", "Opinion", "Religion"]
    },
    {
      "indicator": "More GDELT events involving religious actors making public statements emphasizing the importance of God or religion in society.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["REL"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in GDELT events coded as appeals or calls to religious faith or God by religious leaders or organizations.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["02"],
      "gdelt_actor_types": ["REL"]
    },
    {
      "indicator": "More New York Times articles highlighting political or social movements invoking God or religion as central to American identity or policy.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["God in politics", "faith-based initiatives", "religious values", "Christian right", "moral majority"],
      "nyt_sections": ["U.S.", "Politics", "Opinion"]
    },
    {
      "indicator": "Increase in GDELT events involving religious actors engaging in cooperative or material cooperation activities promoting religious causes or institutions.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["03", "06"],
      "gdelt_actor_types": ["REL"]
    }
  ],
  "B": [
    {
      "indicator": "Increase in New York Times headlines or abstracts discussing secularization, decline of religion, or decreasing importance of God in American life.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["secularization", "decline of religion", "less religious", "God less important", "rise of atheism", "spiritual but not religious"],
      "nyt_sections": ["U.S.", "Opinion", "Religion"]
    },
    {
      "indicator": "More GDELT events involving public statements by secular or atheist groups promoting non-religious worldviews or questioning the role of God in society.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["CVL", "EDU"],
      "gdelt_tone": "more_positive",
      "gdelt_actor_name_contains": ["atheist", "secular", "humanist", "freethinker"]
    },
    {
      "indicator": "Increase in New York Times articles covering controversies or debates about religion losing influence in public life or education.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["religion losing influence", "separation of church and state", "religion in schools", "secular education", "religious decline"],
      "nyt_sections": ["U.S.", "Education", "Opinion"]
    },
    {
      "indicator": "Increase in GDELT protest events involving secular or atheist groups advocating for separation of religion from government or public institutions.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_actor_name_contains": ["secular", "atheist", "humanist"]
    },
    {
      "indicator": "Decrease in GDELT events involving religious actors making public statements or appeals emphasizing God’s importance.",
      "source": "gdelt",
      "expected_change": "decrease",
      "gdelt_event_codes": ["01", "02"],
      "gdelt_actor_types": ["REL"]
    }
  ]
}
```

## 12. F063 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how important God is in people's lives, on a scale from 1 (not at all important) to 10 (very important). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "very important" (religion became more central to people's lives).
Direction B: average answer moved toward "not at all important" (people became more secular, religion less central).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines or abstracts discussing the rise or strengthening of religious faith or God in American life." (source: nyt, predicted: increase)
   Measured: 0/27234 matching articles (0.00 per 1,000) in 2006 vs. 0/19036 (0.00 per 1,000) in 2011 [sections: U.S., Opinion, Religion]
2. "More GDELT events involving religious actors making public statements emphasizing the importance of God or religion in society." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]
3. "Increase in GDELT events coded as appeals or calls to religious faith or God by religious leaders or organizations." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]
4. "More New York Times articles highlighting political or social movements invoking God or religion as central to American identity or policy." (source: nyt, predicted: increase)
   Measured: 5/27234 matching articles (0.18 per 1,000) in 2006 vs. 3/19036 (0.16 per 1,000) in 2011 [sections: U.S., Politics, Opinion]
5. "Increase in GDELT events involving religious actors engaging in cooperative or material cooperation activities promoting religious causes or institutions." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]

=== Direction B ===
1. "Increase in New York Times headlines or abstracts discussing secularization, decline of religion, or decreasing importance of God in American life." (source: nyt, predicted: increase)
   Measured: 0/27234 matching articles (0.00 per 1,000) in 2006 vs. 0/19036 (0.00 per 1,000) in 2011 [sections: U.S., Opinion, Religion]
2. "More GDELT events involving public statements by secular or atheist groups promoting non-religious worldviews or questioning the role of God in society." (source: gdelt, predicted: increase)
   Measured: 1/969651 matching events (0.00 per 1,000) in 2006 vs. 12/10259907 (0.00 per 1,000) in 2011 [actors: CVL, EDU; actor name contains: atheist, secular, humanist, freethinker]; avg tone 9.18 vs. 6.51
3. "Increase in New York Times articles covering controversies or debates about religion losing influence in public life or education." (source: nyt, predicted: increase)
   Measured: 8/27943 matching articles (0.29 per 1,000) in 2006 vs. 3/21331 (0.14 per 1,000) in 2011 [sections: U.S., Education, Opinion]
4. "Increase in GDELT protest events involving secular or atheist groups advocating for separation of religion from government or public institutions." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: CVL; actor name contains: secular, atheist, humanist]
5. "Decrease in GDELT events involving religious actors making public statements or appeals emphasizing God’s importance." (source: gdelt, predicted: decrease)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "need_more_evidence", "reasoning": "The indicators for both directions show almost no change or very minimal activity, with no clear increase in religious importance or secularization in media coverage or public events. The evidence is insufficient to decisively favor either direction.", "requested_evidence": [{"direction": "A", "indicator": "Increase in nationally representative survey data on frequency of religious service attendance", "source": "survey", "expected_change": "increase"}, {"direction": "B", "indicator": "Increase in nationally representative survey data on percentage of people identifying as religiously unaffiliated (e.g., 'nones')", "source": "survey", "expected_change": "increase"}, {"direction": "A", "indicator": "Increase in Google search trends for religious terms or phrases related to God and faith", "source": "google_trends", "expected_change": "increase"}, {"direction": "B", "indicator": "Increase in Google search trends for secular or atheist terms", "source": "google_trends", "expected_change": "increase"}, {"direction": "A", "indicator": "Increase in book publications or sales related to religion or spirituality", "source": "publishing_data", "expected_change": "increase"}]}
```

## 13. F063 -- aggregate_judge round 2 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how important God is in people's lives, on a scale from 1 (not at all important) to 10 (very important). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "very important" (religion became more central to people's lives).
Direction B: average answer moved toward "not at all important" (people became more secular, religion less central).

This is round 2 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines or abstracts discussing the rise or strengthening of religious faith or God in American life." (source: nyt, predicted: increase)
   Measured: 0/27234 matching articles (0.00 per 1,000) in 2006 vs. 0/19036 (0.00 per 1,000) in 2011 [sections: U.S., Opinion, Religion]
2. "More GDELT events involving religious actors making public statements emphasizing the importance of God or religion in society." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]
3. "Increase in GDELT events coded as appeals or calls to religious faith or God by religious leaders or organizations." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]
4. "More New York Times articles highlighting political or social movements invoking God or religion as central to American identity or policy." (source: nyt, predicted: increase)
   Measured: 5/27234 matching articles (0.18 per 1,000) in 2006 vs. 3/19036 (0.16 per 1,000) in 2011 [sections: U.S., Politics, Opinion]
5. "Increase in GDELT events involving religious actors engaging in cooperative or material cooperation activities promoting religious causes or institutions." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]

=== Direction B ===
1. "Increase in New York Times headlines or abstracts discussing secularization, decline of religion, or decreasing importance of God in American life." (source: nyt, predicted: increase)
   Measured: 0/27234 matching articles (0.00 per 1,000) in 2006 vs. 0/19036 (0.00 per 1,000) in 2011 [sections: U.S., Opinion, Religion]
2. "More GDELT events involving public statements by secular or atheist groups promoting non-religious worldviews or questioning the role of God in society." (source: gdelt, predicted: increase)
   Measured: 1/969651 matching events (0.00 per 1,000) in 2006 vs. 12/10259907 (0.00 per 1,000) in 2011 [actors: CVL, EDU; actor name contains: atheist, secular, humanist, freethinker]; avg tone 9.18 vs. 6.51
3. "Increase in New York Times articles covering controversies or debates about religion losing influence in public life or education." (source: nyt, predicted: increase)
   Measured: 8/27943 matching articles (0.29 per 1,000) in 2006 vs. 3/21331 (0.14 per 1,000) in 2011 [sections: U.S., Education, Opinion]
4. "Increase in GDELT protest events involving secular or atheist groups advocating for separation of religion from government or public institutions." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: CVL; actor name contains: secular, atheist, humanist]
5. "Decrease in GDELT events involving religious actors making public statements or appeals emphasizing God’s importance." (source: gdelt, predicted: decrease)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "need_more_evidence", "reasoning": "The indicators for both directions show almost no change or very minimal activity in the predicted directions, with many zero counts and no clear trend. The secularization indicators show a slight increase in public statements by secular groups, but this is very small and not strongly conclusive. More direct measures of public attitudes or broader media coverage might clarify the trend.", "requested_evidence": [{"direction": "A", "indicator": "Increase in nationally representative survey data on frequency of religious service attendance", "source": "survey", "expected_change": "increase"}, {"direction": "B", "indicator": "Increase in nationally representative survey data on percentage of people identifying as religiously unaffiliated (nones)", "source": "survey", "expected_change": "increase"}, {"direction": "A", "indicator": "Increase in New York Times opinion pieces or editorials explicitly endorsing the importance of God or religion in American life", "source": "nyt", "expected_change": "increase", "nyt_keywords": ["God", "religion", "faith", "spirituality"], "nyt_sections": ["Opinion"]}, {"direction": "B", "indicator": "Increase in New York Times opinion pieces or editorials explicitly endorsing secularism or questioning the role of religion in public life", "source": "nyt", "expected_change": "increase", "nyt_keywords": ["secular", "atheist", "non-religious", "separation of church and state"], "nyt_sections": ["Opinion"]}, {"direction": "A", "indicator": "Increase in GDELT events involving religious leaders or organizations organizing or participating in large public religious events or rallies", "source": "gdelt", "expected_change": "increase", "gdelt_event_codes": ["14"], "gdelt_actor_types": ["REL"]}]}
```

## 14. F063 -- aggregate_judge round 3 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how important God is in people's lives, on a scale from 1 (not at all important) to 10 (very important). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "very important" (religion became more central to people's lives).
Direction B: average answer moved toward "not at all important" (people became more secular, religion less central).

This is round 3 of at most 3. This is the final round. You must answer "A" or "B" -- "need_more_evidence" will not be honored.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines or abstracts discussing the rise or strengthening of religious faith or God in American life." (source: nyt, predicted: increase)
   Measured: 0/27234 matching articles (0.00 per 1,000) in 2006 vs. 0/19036 (0.00 per 1,000) in 2011 [sections: U.S., Opinion, Religion]
2. "More GDELT events involving religious actors making public statements emphasizing the importance of God or religion in society." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]
3. "Increase in GDELT events coded as appeals or calls to religious faith or God by religious leaders or organizations." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]
4. "More New York Times articles highlighting political or social movements invoking God or religion as central to American identity or policy." (source: nyt, predicted: increase)
   Measured: 5/27234 matching articles (0.18 per 1,000) in 2006 vs. 3/19036 (0.16 per 1,000) in 2011 [sections: U.S., Politics, Opinion]
5. "Increase in GDELT events involving religious actors engaging in cooperative or material cooperation activities promoting religious causes or institutions." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]
6. "Increase in New York Times opinion pieces or editorials explicitly endorsing the importance of God or religion in American life" (source: nyt, predicted: increase)
   Measured: 236/16492 matching articles (14.31 per 1,000) in 2006 vs. 71/9196 (7.72 per 1,000) in 2011 [sections: Opinion]
7. "Increase in GDELT events involving religious leaders or organizations organizing or participating in large public religious events or rallies" (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]

=== Direction B ===
1. "Increase in New York Times headlines or abstracts discussing secularization, decline of religion, or decreasing importance of God in American life." (source: nyt, predicted: increase)
   Measured: 0/27234 matching articles (0.00 per 1,000) in 2006 vs. 0/19036 (0.00 per 1,000) in 2011 [sections: U.S., Opinion, Religion]
2. "More GDELT events involving public statements by secular or atheist groups promoting non-religious worldviews or questioning the role of God in society." (source: gdelt, predicted: increase)
   Measured: 1/969651 matching events (0.00 per 1,000) in 2006 vs. 12/10259907 (0.00 per 1,000) in 2011 [actors: CVL, EDU; actor name contains: atheist, secular, humanist, freethinker]; avg tone 9.18 vs. 6.51
3. "Increase in New York Times articles covering controversies or debates about religion losing influence in public life or education." (source: nyt, predicted: increase)
   Measured: 8/27943 matching articles (0.29 per 1,000) in 2006 vs. 3/21331 (0.14 per 1,000) in 2011 [sections: U.S., Education, Opinion]
4. "Increase in GDELT protest events involving secular or atheist groups advocating for separation of religion from government or public institutions." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: CVL; actor name contains: secular, atheist, humanist]
5. "Decrease in GDELT events involving religious actors making public statements or appeals emphasizing God’s importance." (source: gdelt, predicted: decrease)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]
6. "Increase in New York Times opinion pieces or editorials explicitly endorsing secularism or questioning the role of religion in public life" (source: nyt, predicted: increase)
   Measured: 35/16492 matching articles (2.12 per 1,000) in 2006 vs. 12/9196 (1.30 per 1,000) in 2011 [sections: Opinion]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "B", "reasoning": "The evidence shows no increase in indicators supporting a rise in the importance of God or religion (Direction A), with many religious-related events and articles remaining at zero or declining. Conversely, although secular indicators are generally low, there is a slight increase in public statements by secular groups and a decrease in opinion pieces endorsing religion, suggesting a modest shift toward secularization (Direction B). Overall, the data favors a move toward less importance of God in people's lives.", "requested_evidence": []}
```

## 15. E221B -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people have personally attended a peaceful, lawful demonstration recently, coded 1 (have done) or 2 (not done). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "have done" (more people recently attended a peaceful demonstration).
Direction B: average answer moved toward "not done" (fewer people recently attended a peaceful demonstration).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines or abstracts mentioning large-scale peaceful demonstrations or protests in the U.S.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["peaceful demonstration", "mass protest", "peaceful rally", "lawful protest", "public demonstration"],
      "nyt_sections": ["U.S.", "National"]
    },
    {
      "indicator": "More GDELT events coded as peaceful protests (CAMEO code 14 PROTEST) involving civilians in U.S. locations.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_location_contains": "United States"
    },
    {
      "indicator": "Increase in New York Times coverage of grassroots or community organizing efforts leading to peaceful demonstrations.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["community organizing", "grassroots movement", "peaceful protest organization", "mobilization for demonstration"],
      "nyt_sections": ["U.S.", "National"]
    },
    {
      "indicator": "More GDELT events involving police presence at peaceful demonstrations without escalation to violence.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["COP", "CVL"],
      "gdelt_location_contains": "United States",
      "gdelt_goldstein": "more_positive"
    },
    {
      "indicator": "Increase in New York Times opinion pieces or editorials supporting the right to peaceful assembly and reporting on recent demonstrations.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["right to assemble", "peaceful protest support", "demonstration rights", "civil liberties demonstration"],
      "nyt_sections": ["Opinion"]
    }
  ],
  "B": [
    {
      "indicator": "Decrease in New York Times headlines or abstracts mentioning peaceful demonstrations or protests in the U.S.",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["peaceful demonstration", "mass protest", "peaceful rally", "lawful protest", "public demonstration"],
      "nyt_sections": ["U.S.", "National"]
    },
    {
      "indicator": "Fewer GDELT events coded as peaceful protests (CAMEO code 14 PROTEST) involving civilians in U.S. locations.",
      "source": "gdelt",
      "expected_change": "decrease",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_location_contains": "United States"
    },
    {
      "indicator": "Decrease in New York Times coverage of grassroots or community organizing efforts leading to peaceful demonstrations.",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["community organizing", "grassroots movement", "peaceful protest organization", "mobilization for demonstration"],
      "nyt_sections": ["U.S.", "National"]
    },
    {
      "indicator": "Fewer GDELT events involving police presence at peaceful demonstrations without escalation to violence.",
      "source": "gdelt",
      "expected_change": "decrease",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["COP", "CVL"],
      "gdelt_location_contains": "United States",
      "gdelt_goldstein": "more_positive"
    },
    {
      "indicator": "Decrease in New York Times opinion pieces or editorials supporting the right to peaceful assembly and reporting on recent demonstrations.",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["right to assemble", "peaceful protest support", "demonstration rights", "civil liberties demonstration"],
      "nyt_sections": ["Opinion"]
    }
  ]
}
```

## 16. E221B -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people have personally attended a peaceful, lawful demonstration recently, coded 1 (have done) or 2 (not done). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "have done" (more people recently attended a peaceful demonstration).
Direction B: average answer moved toward "not done" (fewer people recently attended a peaceful demonstration).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines or abstracts mentioning large-scale peaceful demonstrations or protests in the U.S." (source: nyt, predicted: increase)
   Measured: 0/12293 matching articles (0.00 per 1,000) in 2006 vs. 0/10627 (0.00 per 1,000) in 2011 [sections: U.S., National]
2. "More GDELT events coded as peaceful protests (CAMEO code 14 PROTEST) involving civilians in U.S. locations." (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL; location contains: U, n, i, t, e, d,  , S, t, a, t, e, s]
3. "Increase in New York Times coverage of grassroots or community organizing efforts leading to peaceful demonstrations." (source: nyt, predicted: increase)
   Measured: 1/12293 matching articles (0.08 per 1,000) in 2006 vs. 0/10627 (0.00 per 1,000) in 2011 [sections: U.S., National]
4. "More GDELT events involving police presence at peaceful demonstrations without escalation to violence." (source: gdelt, predicted: increase)
   Measured: 616/969651 matching events (0.64 per 1,000) in 2006 vs. 8722/10259907 (0.85 per 1,000) in 2011 [actors: COP, CVL; location contains: U, n, i, t, e, d,  , S, t, a, t, e, s]; avg Goldstein -6.69 vs. -6.69
5. "Increase in New York Times opinion pieces or editorials supporting the right to peaceful assembly and reporting on recent demonstrations." (source: nyt, predicted: increase)
   Measured: 0/16492 matching articles (0.00 per 1,000) in 2006 vs. 1/9196 (0.11 per 1,000) in 2011 [sections: Opinion]

=== Direction B ===
1. "Decrease in New York Times headlines or abstracts mentioning peaceful demonstrations or protests in the U.S." (source: nyt, predicted: decrease)
   Measured: 0/12293 matching articles (0.00 per 1,000) in 2006 vs. 0/10627 (0.00 per 1,000) in 2011 [sections: U.S., National]
2. "Fewer GDELT events coded as peaceful protests (CAMEO code 14 PROTEST) involving civilians in U.S. locations." (source: gdelt, predicted: decrease)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL; location contains: U, n, i, t, e, d,  , S, t, a, t, e, s]
3. "Decrease in New York Times coverage of grassroots or community organizing efforts leading to peaceful demonstrations." (source: nyt, predicted: decrease)
   Measured: 1/12293 matching articles (0.08 per 1,000) in 2006 vs. 0/10627 (0.00 per 1,000) in 2011 [sections: U.S., National]
4. "Fewer GDELT events involving police presence at peaceful demonstrations without escalation to violence." (source: gdelt, predicted: decrease)
   Measured: 616/969651 matching events (0.64 per 1,000) in 2006 vs. 8722/10259907 (0.85 per 1,000) in 2011 [actors: COP, CVL; location contains: U, n, i, t, e, d,  , S, t, a, t, e, s]; avg Goldstein -6.69 vs. -6.69
5. "Decrease in New York Times opinion pieces or editorials supporting the right to peaceful assembly and reporting on recent demonstrations." (source: nyt, predicted: decrease)
   Measured: 0/16492 matching articles (0.00 per 1,000) in 2006 vs. 1/9196 (0.11 per 1,000) in 2011 [sections: Opinion]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "A", "reasoning": "The GDELT data shows a clear increase in peaceful protest events (from 0.28 to 0.36 per 1,000 events) and police presence at peaceful demonstrations (from 0.64 to 0.85 per 1,000 events) between 2006 and 2011, supporting Direction A. Although New York Times coverage of peaceful demonstrations is minimal and unchanged, the increase in actual protest events and police presence without violence is a stronger and more direct indicator of increased attendance at peaceful demonstrations.", "requested_evidence": []}
```

## 17. F121 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how justifiable people consider divorce, on a scale from 1 (never justifiable) to 10 (always justifiable). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "always justifiable" (attitudes became more accepting of divorce).
Direction B: average answer moved toward "never justifiable" (attitudes became less accepting of divorce).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines discussing divorce as a common or acceptable solution to marital problems",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["divorce", "acceptable", "common", "justifiable", "marital problems"],
      "nyt_sections": ["U.S.", "Opinion", "Style"]
    },
    {
      "indicator": "More GDELT events involving public figures or government officials making statements supporting divorce rights or reforms to make divorce easier",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["GOV", "MED"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in NYT articles highlighting social acceptance of divorce, including stories about families or individuals positively framing divorce",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["divorce", "acceptance", "family", "positive", "new beginnings"],
      "nyt_sections": ["U.S.", "Style"]
    },
    {
      "indicator": "More GDELT protest or appeal events related to divorce law liberalization or women's rights to divorce",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["02", "14"],
      "gdelt_actor_types": ["CVL", "REL", "GOV"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in NYT opinion pieces or editorials advocating for easier divorce procedures or recognizing divorce as a justified choice",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["divorce", "justified", "opinion", "editorial", "reform"],
      "nyt_sections": ["Opinion"]
    }
  ],
  "B": [
    {
      "indicator": "Increase in New York Times headlines emphasizing the negative consequences of divorce or promoting marriage preservation",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["divorce", "negative consequences", "marriage preservation", "family values", "harmful"],
      "nyt_sections": ["U.S.", "Opinion"]
    },
    {
      "indicator": "More GDELT events involving religious leaders or organizations publicly condemning divorce or advocating for stronger marriage laws",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["REL"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in NYT articles reporting on government or legislative efforts to restrict divorce access or make divorce laws more stringent",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["divorce law", "restrict", "legislation", "marriage protection", "divorce reform"],
      "nyt_sections": ["U.S."]
    },
    {
      "indicator": "More GDELT events showing public protests or appeals against divorce liberalization or in favor of traditional marriage values",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["02", "14"],
      "gdelt_actor_types": ["CVL", "REL"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in NYT opinion pieces or editorials arguing that divorce is harmful or should be less justifiable socially",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["divorce", "harmful", "opinion", "editorial", "marriage"],
      "nyt_sections": ["Opinion"]
    }
  ]
}
```

## 18. F121 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how justifiable people consider divorce, on a scale from 1 (never justifiable) to 10 (always justifiable). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "always justifiable" (attitudes became more accepting of divorce).
Direction B: average answer moved toward "never justifiable" (attitudes became less accepting of divorce).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines discussing divorce as a common or acceptable solution to marital problems" (source: nyt, predicted: increase)
   Measured: 410/30747 matching articles (13.33 per 1,000) in 2006 vs. 109/20881 (5.22 per 1,000) in 2011 [sections: U.S., Opinion, Style]
2. "More GDELT events involving public figures or government officials making statements supporting divorce rights or reforms to make divorce easier" (source: gdelt, predicted: increase)
   Measured: 45641/969651 matching events (47.07 per 1,000) in 2006 vs. 371666/10259907 (36.23 per 1,000) in 2011 [actors: GOV, MED]; avg tone 5.60 vs. 5.46
3. "Increase in NYT articles highlighting social acceptance of divorce, including stories about families or individuals positively framing divorce" (source: nyt, predicted: increase)
   Measured: 462/14255 matching articles (32.41 per 1,000) in 2006 vs. 203/11685 (17.37 per 1,000) in 2011 [sections: U.S., Style]
4. "More GDELT protest or appeal events related to divorce law liberalization or women's rights to divorce" (source: gdelt, predicted: increase)
   Measured: 20862/969651 matching events (21.51 per 1,000) in 2006 vs. 230121/10259907 (22.43 per 1,000) in 2011 [actors: CVL, REL, GOV]; avg tone 5.97 vs. 5.83
5. "Increase in NYT opinion pieces or editorials advocating for easier divorce procedures or recognizing divorce as a justified choice" (source: nyt, predicted: increase)
   Measured: 1945/16492 matching articles (117.94 per 1,000) in 2006 vs. 462/9196 (50.24 per 1,000) in 2011 [sections: Opinion]

=== Direction B ===
1. "Increase in New York Times headlines emphasizing the negative consequences of divorce or promoting marriage preservation" (source: nyt, predicted: increase)
   Measured: 67/27234 matching articles (2.46 per 1,000) in 2006 vs. 31/19036 (1.63 per 1,000) in 2011 [sections: U.S., Opinion]
2. "More GDELT events involving religious leaders or organizations publicly condemning divorce or advocating for stronger marriage laws" (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL]
3. "Increase in NYT articles reporting on government or legislative efforts to restrict divorce access or make divorce laws more stringent" (source: nyt, predicted: increase)
   Measured: 272/10742 matching articles (25.32 per 1,000) in 2006 vs. 110/9840 (11.18 per 1,000) in 2011 [sections: U.S.]
4. "More GDELT events showing public protests or appeals against divorce liberalization or in favor of traditional marriage values" (source: gdelt, predicted: increase)
   Measured: 3900/969651 matching events (4.02 per 1,000) in 2006 vs. 68674/10259907 (6.69 per 1,000) in 2011 [actors: CVL, REL]; avg tone 6.10 vs. 5.85
5. "Increase in NYT opinion pieces or editorials arguing that divorce is harmful or should be less justifiable socially" (source: nyt, predicted: increase)
   Measured: 1790/16492 matching articles (108.54 per 1,000) in 2006 vs. 381/9196 (41.43 per 1,000) in 2011 [sections: Opinion]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "B", "reasoning": "All indicators for Direction A show a decrease rather than the predicted increase, including fewer NYT articles and GDELT events supporting divorce acceptance. Conversely, most indicators for Direction B also show a decrease, but the increase in protests or appeals against divorce liberalization and the presence of opinion pieces arguing divorce is harmful support a shift toward less acceptance. Overall, the evidence more strongly supports Direction B.", "requested_evidence": []}
```

## 19. E026B -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people have personally joined in a boycott recently, coded 1 (have done) or 2 (not done). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "have done" (more people recently joined in a boycott).
Direction B: average answer moved toward "not done" (fewer people recently joined in a boycott).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines or abstracts mentioning consumer boycotts or calls for boycotts in the U.S.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["boycott", "consumer boycott", "buycott", "protest buying"],
      "nyt_sections": ["U.S.", "Business", "Opinion"]
    },
    {
      "indicator": "More GDELT events coded as '14 PROTEST' involving civilians or business actors related to boycotts or consumer activism in the U.S.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL", "BUS"],
      "gdelt_location_contains": ["United States", "USA", "U.S."],
      "gdelt_baseline_codes": ["01", "02", "10", "11", "14"]
    },
    {
      "indicator": "Increase in New York Times coverage of organized boycott campaigns targeting specific companies or industries in the U.S.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["boycott campaign", "corporate boycott", "consumer activism", "boycott movement"],
      "nyt_sections": ["Business", "U.S."]
    },
    {
      "indicator": "More GDELT events with '10 DEMAND' or '11 DISAPPROVE' codes involving consumer groups or activists demanding boycotts or expressing disapproval of companies.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["10", "11"],
      "gdelt_actor_types": ["CVL", "BUS"],
      "gdelt_location_contains": ["United States"],
      "gdelt_baseline_codes": ["01", "02", "10", "11", "14"]
    },
    {
      "indicator": "Increase in New York Times opinion pieces or editorials encouraging or discussing the effectiveness of boycotts as a form of protest.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["boycott effectiveness", "boycott opinion", "consumer protest opinion"],
      "nyt_sections": ["Opinion"]
    }
  ],
  "B": [
    {
      "indicator": "Decrease in New York Times headlines or abstracts mentioning consumer boycotts or calls for boycotts in the U.S.",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["boycott", "consumer boycott", "buycott", "protest buying"],
      "nyt_sections": ["U.S.", "Business", "Opinion"]
    },
    {
      "indicator": "Fewer GDELT events coded as '14 PROTEST' involving civilians or business actors related to boycotts or consumer activism in the U.S.",
      "source": "gdelt",
      "expected_change": "decrease",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL", "BUS"],
      "gdelt_location_contains": ["United States", "USA", "U.S."],
      "gdelt_baseline_codes": ["01", "02", "10", "11", "14"]
    },
    {
      "indicator": "Decrease in New York Times coverage of organized boycott campaigns targeting specific companies or industries in the U.S.",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["boycott campaign", "corporate boycott", "consumer activism", "boycott movement"],
      "nyt_sections": ["Business", "U.S."]
    },
    {
      "indicator": "Fewer GDELT events with '10 DEMAND' or '11 DISAPPROVE' codes involving consumer groups or activists demanding boycotts or expressing disapproval of companies.",
      "source": "gdelt",
      "expected_change": "decrease",
      "gdelt_event_codes": ["10", "11"],
      "gdelt_actor_types": ["CVL", "BUS"],
      "gdelt_location_contains": ["United States"],
      "gdelt_baseline_codes": ["01", "02", "10", "11", "14"]
    },
    {
      "indicator": "Decrease in New York Times opinion pieces or editorials encouraging or discussing the effectiveness of boycotts as a form of protest.",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["boycott effectiveness", "boycott opinion", "consumer protest opinion"],
      "nyt_sections": ["Opinion"]
    }
  ]
}
```

## 20. E026B -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people have personally joined in a boycott recently, coded 1 (have done) or 2 (not done). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "have done" (more people recently joined in a boycott).
Direction B: average answer moved toward "not done" (fewer people recently joined in a boycott).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines or abstracts mentioning consumer boycotts or calls for boycotts in the U.S." (source: nyt, predicted: increase)
   Measured: 37/42512 matching articles (0.87 per 1,000) in 2006 vs. 11/25239 (0.44 per 1,000) in 2011 [sections: U.S., Business, Opinion]
2. "More GDELT events coded as '14 PROTEST' involving civilians or business actors related to boycotts or consumer activism in the U.S." (source: gdelt, predicted: increase)
   Measured: 540/320368 matching events (1.69 per 1,000) in 2006 vs. 7390/3125364 (2.36 per 1,000) in 2011 [actors: CVL, BUS; location contains: United States, USA, U.S.; rate vs. baseline categories 01, 02, 10, 11, 14 (not all events)]
3. "Increase in New York Times coverage of organized boycott campaigns targeting specific companies or industries in the U.S." (source: nyt, predicted: increase)
   Measured: 0/26020 matching articles (0.00 per 1,000) in 2006 vs. 0/16043 (0.00 per 1,000) in 2011 [sections: Business, U.S.]
4. "More GDELT events with '10 DEMAND' or '11 DISAPPROVE' codes involving consumer groups or activists demanding boycotts or expressing disapproval of companies." (source: gdelt, predicted: increase)
   Measured: 9637/320368 matching events (30.08 per 1,000) in 2006 vs. 95583/3125364 (30.58 per 1,000) in 2011 [actors: CVL, BUS; location contains: United States; rate vs. baseline categories 01, 02, 10, 11, 14 (not all events)]
5. "Increase in New York Times opinion pieces or editorials encouraging or discussing the effectiveness of boycotts as a form of protest." (source: nyt, predicted: increase)
   Measured: 0/16492 matching articles (0.00 per 1,000) in 2006 vs. 0/9196 (0.00 per 1,000) in 2011 [sections: Opinion]

=== Direction B ===
1. "Decrease in New York Times headlines or abstracts mentioning consumer boycotts or calls for boycotts in the U.S." (source: nyt, predicted: decrease)
   Measured: 37/42512 matching articles (0.87 per 1,000) in 2006 vs. 11/25239 (0.44 per 1,000) in 2011 [sections: U.S., Business, Opinion]
2. "Fewer GDELT events coded as '14 PROTEST' involving civilians or business actors related to boycotts or consumer activism in the U.S." (source: gdelt, predicted: decrease)
   Measured: 540/320368 matching events (1.69 per 1,000) in 2006 vs. 7390/3125364 (2.36 per 1,000) in 2011 [actors: CVL, BUS; location contains: United States, USA, U.S.; rate vs. baseline categories 01, 02, 10, 11, 14 (not all events)]
3. "Decrease in New York Times coverage of organized boycott campaigns targeting specific companies or industries in the U.S." (source: nyt, predicted: decrease)
   Measured: 0/26020 matching articles (0.00 per 1,000) in 2006 vs. 0/16043 (0.00 per 1,000) in 2011 [sections: Business, U.S.]
4. "Fewer GDELT events with '10 DEMAND' or '11 DISAPPROVE' codes involving consumer groups or activists demanding boycotts or expressing disapproval of companies." (source: gdelt, predicted: decrease)
   Measured: 9637/320368 matching events (30.08 per 1,000) in 2006 vs. 95583/3125364 (30.58 per 1,000) in 2011 [actors: CVL, BUS; location contains: United States; rate vs. baseline categories 01, 02, 10, 11, 14 (not all events)]
5. "Decrease in New York Times opinion pieces or editorials encouraging or discussing the effectiveness of boycotts as a form of protest." (source: nyt, predicted: decrease)
   Measured: 0/16492 matching articles (0.00 per 1,000) in 2006 vs. 0/9196 (0.00 per 1,000) in 2011 [sections: Opinion]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "B", "reasoning": "The evidence strongly supports Direction B, showing a clear decrease in New York Times mentions of boycotts and no increase in opinion pieces or organized boycott coverage. Although GDELT protest events related to boycotts increased, the lack of media coverage and opinion support suggests less public engagement in boycotts. The large decrease in media mentions is a strong indicator that fewer people joined boycotts.", "requested_evidence": []}
```

## 21. E217 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people agree that science and technology are making our lives healthier, easier and more comfortable, on a scale from 1 (completely disagree) to 10 (completely agree). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "completely agree" (people became more positive about the effect of science and technology on daily life).
Direction B: average answer moved toward "completely disagree" (people became more negative about the effect of science and technology on daily life).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in positive news headlines about breakthroughs in medical technology improving health outcomes",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["medical technology breakthrough", "health innovation", "new medical device", "cure", "health improvement"],
      "nyt_sections": ["Health", "Science"]
    },
    {
      "indicator": "More GDELT events reporting government or business announcements of new technology initiatives aimed at improving quality of life",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["GOV", "BUS"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in NYT articles highlighting the benefits of technology in making daily tasks easier and more comfortable",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["technology makes life easier", "technology convenience", "smart home", "automation benefits"],
      "nyt_sections": ["Technology", "U.S."]
    },
    {
      "indicator": "More GDELT events involving educational institutions promoting STEM education and technological literacy",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["03", "05"],
      "gdelt_actor_types": ["EDU"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in NYT coverage of successful tech startups and innovations contributing to economic growth and improved lifestyles",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["tech startup success", "innovation economy", "technology improves life", "economic growth technology"],
      "nyt_sections": ["Business", "Technology"]
    }
  ],
  "B": [
    {
      "indicator": "Increase in NYT headlines expressing skepticism or criticism of technology's impact on health and comfort",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["technology harms health", "technology causes problems", "technology backlash", "negative impact of technology"],
      "nyt_sections": ["Opinion", "Science", "Health"]
    },
    {
      "indicator": "More GDELT events reporting protests or public disapproval against technological developments perceived as harmful or intrusive",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14", "11"],
      "gdelt_actor_types": ["CVL", "MED"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in NYT articles covering technology-related privacy concerns, data breaches, or ethical controversies",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["privacy concerns", "data breach", "technology ethics", "surveillance technology"],
      "nyt_sections": ["Technology", "U.S."]
    },
    {
      "indicator": "More GDELT events involving government investigations or regulatory actions against technology companies for negative social impacts",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["09", "10"],
      "gdelt_actor_types": ["GOV", "JUD"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in NYT coverage of technology failures or accidents causing harm or discomfort to people",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["technology failure", "accident caused by technology", "technology malfunction", "harmful technology"],
      "nyt_sections": ["U.S.", "Science"]
    }
  ]
}
```

## 22. E217 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people agree that science and technology are making our lives healthier, easier and more comfortable, on a scale from 1 (completely disagree) to 10 (completely agree). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "completely agree" (people became more positive about the effect of science and technology on daily life).
Direction B: average answer moved toward "completely disagree" (people became more negative about the effect of science and technology on daily life).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in positive news headlines about breakthroughs in medical technology improving health outcomes" (source: nyt, predicted: increase)
   Measured: 19/2887 matching articles (6.58 per 1,000) in 2006 vs. 25/4503 (5.55 per 1,000) in 2011 [sections: Health, Science]
2. "More GDELT events reporting government or business announcements of new technology initiatives aimed at improving quality of life" (source: gdelt, predicted: increase)
   Measured: 45088/969651 matching events (46.50 per 1,000) in 2006 vs. 353390/10259907 (34.44 per 1,000) in 2011 [actors: GOV, BUS]; avg tone 6.19 vs. 5.72
3. "Increase in NYT articles highlighting the benefits of technology in making daily tasks easier and more comfortable" (source: nyt, predicted: increase)
   Measured: 0/12698 matching articles (0.00 per 1,000) in 2006 vs. 0/12562 (0.00 per 1,000) in 2011 [sections: Technology, U.S.]
4. "More GDELT events involving educational institutions promoting STEM education and technological literacy" (source: gdelt, predicted: increase)
   Measured: 4818/969651 matching events (4.97 per 1,000) in 2006 vs. 149069/10259907 (14.53 per 1,000) in 2011 [actors: EDU]; avg tone 7.33 vs. 7.47
5. "Increase in NYT coverage of successful tech startups and innovations contributing to economic growth and improved lifestyles" (source: nyt, predicted: increase)
   Measured: 0/16175 matching articles (0.00 per 1,000) in 2006 vs. 0/8117 (0.00 per 1,000) in 2011 [sections: Business, Technology]

=== Direction B ===
1. "Increase in NYT headlines expressing skepticism or criticism of technology's impact on health and comfort" (source: nyt, predicted: increase)
   Measured: 0/19379 matching articles (0.00 per 1,000) in 2006 vs. 0/13699 (0.00 per 1,000) in 2011 [sections: Opinion, Science, Health]
2. "More GDELT events reporting protests or public disapproval against technological developments perceived as harmful or intrusive" (source: gdelt, predicted: increase)
   Measured: 6272/969651 matching events (6.47 per 1,000) in 2006 vs. 72274/10259907 (7.04 per 1,000) in 2011 [actors: CVL, MED]; avg tone 5.30 vs. 5.27
3. "Increase in NYT articles covering technology-related privacy concerns, data breaches, or ethical controversies" (source: nyt, predicted: increase)
   Measured: 7/12698 matching articles (0.55 per 1,000) in 2006 vs. 11/12562 (0.88 per 1,000) in 2011 [sections: Technology, U.S.]
4. "More GDELT events involving government investigations or regulatory actions against technology companies for negative social impacts" (source: gdelt, predicted: increase)
   Measured: 9571/969651 matching events (9.87 per 1,000) in 2006 vs. 86082/10259907 (8.39 per 1,000) in 2011 [actors: GOV, JUD]; avg tone 5.24 vs. 4.82
5. "Increase in NYT coverage of technology failures or accidents causing harm or discomfort to people" (source: nyt, predicted: increase)
   Measured: 0/12074 matching articles (0.00 per 1,000) in 2006 vs. 0/13041 (0.00 per 1,000) in 2011 [sections: U.S., Science]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision":"need_more_evidence","reasoning":"The evidence for Direction A shows some increase in educational events promoting STEM but a decrease or no change in positive news coverage about technology benefits. Direction B shows slight increases in protests and privacy concerns but no increase in skepticism or criticism headlines. The indicators are mixed and mostly show small or no changes, so more direct measures of public sentiment or broader media tone are needed to decide.","requested_evidence":[{"direction":"A","indicator":"Overall public opinion polls on trust in science and technology","source":"polling","expected_change":"increase"},{"direction":"B","indicator":"Increase in NYT opinion pieces or editorials critical of technology's societal impact","source":"nyt","expected_change":"increase","nyt_sections":["Opinion","Science","Technology"]},{"direction":"A","indicator":"Increase in positive social media sentiment about technology improving daily life","source":"social_media","expected_change":"increase"},{"direction":"B","indicator":"Increase in GDELT events related to protests or civil actions against technology companies or innovations","source":"gdelt","expected_change":"increase","gdelt_event_codes":["14"],"gdelt_actor_types":["CVL"]},{"direction":"A","indicator":"Increase in NYT feature articles on technological innovations improving health and comfort","source":"nyt","expected_change":"increase","nyt_sections":["Health","Science","Technology"]}]}
```

## 23. E217 -- aggregate_judge round 2 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people agree that science and technology are making our lives healthier, easier and more comfortable, on a scale from 1 (completely disagree) to 10 (completely agree). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "completely agree" (people became more positive about the effect of science and technology on daily life).
Direction B: average answer moved toward "completely disagree" (people became more negative about the effect of science and technology on daily life).

This is round 2 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in positive news headlines about breakthroughs in medical technology improving health outcomes" (source: nyt, predicted: increase)
   Measured: 19/2887 matching articles (6.58 per 1,000) in 2006 vs. 25/4503 (5.55 per 1,000) in 2011 [sections: Health, Science]
2. "More GDELT events reporting government or business announcements of new technology initiatives aimed at improving quality of life" (source: gdelt, predicted: increase)
   Measured: 45088/969651 matching events (46.50 per 1,000) in 2006 vs. 353390/10259907 (34.44 per 1,000) in 2011 [actors: GOV, BUS]; avg tone 6.19 vs. 5.72
3. "Increase in NYT articles highlighting the benefits of technology in making daily tasks easier and more comfortable" (source: nyt, predicted: increase)
   Measured: 0/12698 matching articles (0.00 per 1,000) in 2006 vs. 0/12562 (0.00 per 1,000) in 2011 [sections: Technology, U.S.]
4. "More GDELT events involving educational institutions promoting STEM education and technological literacy" (source: gdelt, predicted: increase)
   Measured: 4818/969651 matching events (4.97 per 1,000) in 2006 vs. 149069/10259907 (14.53 per 1,000) in 2011 [actors: EDU]; avg tone 7.33 vs. 7.47
5. "Increase in NYT coverage of successful tech startups and innovations contributing to economic growth and improved lifestyles" (source: nyt, predicted: increase)
   Measured: 0/16175 matching articles (0.00 per 1,000) in 2006 vs. 0/8117 (0.00 per 1,000) in 2011 [sections: Business, Technology]

=== Direction B ===
1. "Increase in NYT headlines expressing skepticism or criticism of technology's impact on health and comfort" (source: nyt, predicted: increase)
   Measured: 0/19379 matching articles (0.00 per 1,000) in 2006 vs. 0/13699 (0.00 per 1,000) in 2011 [sections: Opinion, Science, Health]
2. "More GDELT events reporting protests or public disapproval against technological developments perceived as harmful or intrusive" (source: gdelt, predicted: increase)
   Measured: 6272/969651 matching events (6.47 per 1,000) in 2006 vs. 72274/10259907 (7.04 per 1,000) in 2011 [actors: CVL, MED]; avg tone 5.30 vs. 5.27
3. "Increase in NYT articles covering technology-related privacy concerns, data breaches, or ethical controversies" (source: nyt, predicted: increase)
   Measured: 7/12698 matching articles (0.55 per 1,000) in 2006 vs. 11/12562 (0.88 per 1,000) in 2011 [sections: Technology, U.S.]
4. "More GDELT events involving government investigations or regulatory actions against technology companies for negative social impacts" (source: gdelt, predicted: increase)
   Measured: 9571/969651 matching events (9.87 per 1,000) in 2006 vs. 86082/10259907 (8.39 per 1,000) in 2011 [actors: GOV, JUD]; avg tone 5.24 vs. 4.82
5. "Increase in NYT coverage of technology failures or accidents causing harm or discomfort to people" (source: nyt, predicted: increase)
   Measured: 0/12074 matching articles (0.00 per 1,000) in 2006 vs. 0/13041 (0.00 per 1,000) in 2011 [sections: U.S., Science]
6. "Increase in GDELT events related to protests or civil actions against technology companies or innovations" (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "need_more_evidence", "reasoning": "The evidence for Direction A shows some decreases or no change in positive news coverage and events, while Direction B shows slight increases in protests and privacy concerns but also some decreases in regulatory actions and no increase in critical headlines. The indicators are mixed and mostly show small changes, so more direct measures of public sentiment or broader media tone are needed.", "requested_evidence": [{"direction": "A", "indicator": "Overall public opinion polls on trust in science and technology improving daily life", "source": "survey", "expected_change": "increase"}, {"direction": "B", "indicator": "Increase in opinion pieces or editorials expressing skepticism about technology's societal benefits", "source": "nyt", "expected_change": "increase", "nyt_sections": ["Opinion", "Science"]}, {"direction": "A", "indicator": "Increase in positive social media sentiment about technology improving quality of life", "source": "social_media", "expected_change": "increase"}, {"direction": "B", "indicator": "Increase in GDELT events coded as protests or disapprovals specifically targeting technology companies", "source": "gdelt", "expected_change": "increase", "gdelt_event_codes": ["14", "11"], "gdelt_actor_types": ["CVL", "BUS"]}, {"direction": "A", "indicator": "Increase in educational campaigns or government initiatives promoting benefits of technology", "source": "gdelt", "expected_change": "increase", "gdelt_event_codes": ["01", "03"], "gdelt_actor_types": ["GOV", "EDU"]}]}
```

## 24. E217 -- aggregate_judge round 3 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people agree that science and technology are making our lives healthier, easier and more comfortable, on a scale from 1 (completely disagree) to 10 (completely agree). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "completely agree" (people became more positive about the effect of science and technology on daily life).
Direction B: average answer moved toward "completely disagree" (people became more negative about the effect of science and technology on daily life).

This is round 3 of at most 3. This is the final round. You must answer "A" or "B" -- "need_more_evidence" will not be honored.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in positive news headlines about breakthroughs in medical technology improving health outcomes" (source: nyt, predicted: increase)
   Measured: 19/2887 matching articles (6.58 per 1,000) in 2006 vs. 25/4503 (5.55 per 1,000) in 2011 [sections: Health, Science]
2. "More GDELT events reporting government or business announcements of new technology initiatives aimed at improving quality of life" (source: gdelt, predicted: increase)
   Measured: 45088/969651 matching events (46.50 per 1,000) in 2006 vs. 353390/10259907 (34.44 per 1,000) in 2011 [actors: GOV, BUS]; avg tone 6.19 vs. 5.72
3. "Increase in NYT articles highlighting the benefits of technology in making daily tasks easier and more comfortable" (source: nyt, predicted: increase)
   Measured: 0/12698 matching articles (0.00 per 1,000) in 2006 vs. 0/12562 (0.00 per 1,000) in 2011 [sections: Technology, U.S.]
4. "More GDELT events involving educational institutions promoting STEM education and technological literacy" (source: gdelt, predicted: increase)
   Measured: 4818/969651 matching events (4.97 per 1,000) in 2006 vs. 149069/10259907 (14.53 per 1,000) in 2011 [actors: EDU]; avg tone 7.33 vs. 7.47
5. "Increase in NYT coverage of successful tech startups and innovations contributing to economic growth and improved lifestyles" (source: nyt, predicted: increase)
   Measured: 0/16175 matching articles (0.00 per 1,000) in 2006 vs. 0/8117 (0.00 per 1,000) in 2011 [sections: Business, Technology]
6. "Increase in educational campaigns or government initiatives promoting benefits of technology" (source: gdelt, predicted: increase)
   Measured: 54281/969651 matching events (55.98 per 1,000) in 2006 vs. 549357/10259907 (53.54 per 1,000) in 2011 [actors: GOV, EDU]

=== Direction B ===
1. "Increase in NYT headlines expressing skepticism or criticism of technology's impact on health and comfort" (source: nyt, predicted: increase)
   Measured: 0/19379 matching articles (0.00 per 1,000) in 2006 vs. 0/13699 (0.00 per 1,000) in 2011 [sections: Opinion, Science, Health]
2. "More GDELT events reporting protests or public disapproval against technological developments perceived as harmful or intrusive" (source: gdelt, predicted: increase)
   Measured: 6272/969651 matching events (6.47 per 1,000) in 2006 vs. 72274/10259907 (7.04 per 1,000) in 2011 [actors: CVL, MED]; avg tone 5.30 vs. 5.27
3. "Increase in NYT articles covering technology-related privacy concerns, data breaches, or ethical controversies" (source: nyt, predicted: increase)
   Measured: 7/12698 matching articles (0.55 per 1,000) in 2006 vs. 11/12562 (0.88 per 1,000) in 2011 [sections: Technology, U.S.]
4. "More GDELT events involving government investigations or regulatory actions against technology companies for negative social impacts" (source: gdelt, predicted: increase)
   Measured: 9571/969651 matching events (9.87 per 1,000) in 2006 vs. 86082/10259907 (8.39 per 1,000) in 2011 [actors: GOV, JUD]; avg tone 5.24 vs. 4.82
5. "Increase in NYT coverage of technology failures or accidents causing harm or discomfort to people" (source: nyt, predicted: increase)
   Measured: 0/12074 matching articles (0.00 per 1,000) in 2006 vs. 0/13041 (0.00 per 1,000) in 2011 [sections: U.S., Science]
6. "Increase in GDELT events related to protests or civil actions against technology companies or innovations" (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL]
7. "Increase in GDELT events coded as protests or disapprovals specifically targeting technology companies" (source: gdelt, predicted: increase)
   Measured: 8805/969651 matching events (9.08 per 1,000) in 2006 vs. 85024/10259907 (8.29 per 1,000) in 2011 [actors: CVL, BUS]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "B", "reasoning": "The indicators for Direction B show slight increases in protests, public disapproval, and coverage of privacy concerns related to technology, while Direction A indicators mostly show decreases or no change in positive coverage and initiatives. The negative tone and protest-related events against technology companies increased or remained stable, suggesting a shift toward more negative attitudes about technology's impact on daily life.", "requested_evidence": []}
```

## 25. E037 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people think individuals should take more responsibility for providing for themselves, or the government should take more responsibility for ensuring everyone is provided for, on a scale from 1 (people should take more responsibility) to 10 (the government should take more responsibility). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "the government should take more responsibility" (people became more supportive of government responsibility for provision).
Direction B: average answer moved toward "people should take more responsibility" (people became more supportive of individual responsibility for provision).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines emphasizing government programs expanding social safety nets and welfare support.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["government responsibility", "social safety net", "welfare expansion", "public assistance", "government aid"],
      "nyt_sections": ["U.S.", "Politics", "Opinion"]
    },
    {
      "indicator": "More GDELT events coded as government PROVIDE AID (07) related to social welfare programs and public assistance initiatives.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["07"],
      "gdelt_actor_types": ["GOV"],
      "gdelt_location_contains": ["United States"]
    },
    {
      "indicator": "Increase in New York Times articles discussing government interventions in economic crises, such as stimulus packages or unemployment benefits.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["government stimulus", "unemployment benefits", "economic intervention", "public funding", "government bailout"],
      "nyt_sections": ["Business", "U.S.", "Politics"]
    },
    {
      "indicator": "More GDELT events involving government officials making public statements (01) advocating for increased government responsibility in social welfare.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["GOV"],
      "gdelt_location_contains": ["United States"]
    },
    {
      "indicator": "Increase in New York Times opinion pieces and editorials supporting expanded government roles in healthcare, education, and social services.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["government responsibility", "public healthcare", "education funding", "social services", "government role"],
      "nyt_sections": ["Opinion"]
    }
  ],
  "B": [
    {
      "indicator": "Increase in New York Times headlines emphasizing personal responsibility, self-reliance, and critiques of government dependency.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["personal responsibility", "self-reliance", "government dependency", "individual initiative", "welfare reform"],
      "nyt_sections": ["U.S.", "Opinion", "Politics"]
    },
    {
      "indicator": "More GDELT events coded as government DISAPPROVE (11) or REJECT (12) related to expanding welfare programs or government aid.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["11", "12"],
      "gdelt_actor_types": ["GOV", "LEG"],
      "gdelt_location_contains": ["United States"]
    },
    {
      "indicator": "Increase in New York Times articles covering political debates or legislation aimed at reducing government welfare programs and promoting work requirements.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["welfare reform", "work requirements", "reduce government aid", "cut social programs", "personal responsibility"],
      "nyt_sections": ["Politics", "U.S."]
    },
    {
      "indicator": "More GDELT events involving protests (14) or public statements (01) by civilians or politicians advocating for reduced government intervention and increased individual responsibility.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14", "01"],
      "gdelt_actor_types": ["CVL", "GOV", "LEG"],
      "gdelt_location_contains": ["United States"]
    },
    {
      "indicator": "Increase in New York Times opinion pieces criticizing government overreach and promoting free-market solutions and individual accountability.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["government overreach", "free market", "individual accountability", "limited government", "personal responsibility"],
      "nyt_sections": ["Opinion"]
    }
  ]
}
```

## 26. E037 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people think individuals should take more responsibility for providing for themselves, or the government should take more responsibility for ensuring everyone is provided for, on a scale from 1 (people should take more responsibility) to 10 (the government should take more responsibility). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "the government should take more responsibility" (people became more supportive of government responsibility for provision).
Direction B: average answer moved toward "people should take more responsibility" (people became more supportive of individual responsibility for provision).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines emphasizing government programs expanding social safety nets and welfare support." (source: nyt, predicted: increase)
   Measured: 4/27234 matching articles (0.15 per 1,000) in 2006 vs. 3/19036 (0.16 per 1,000) in 2011 [sections: U.S., Politics, Opinion]
2. "More GDELT events coded as government PROVIDE AID (07) related to social welfare programs and public assistance initiatives." (source: gdelt, predicted: increase)
   Measured: 4433/969651 matching events (4.57 per 1,000) in 2006 vs. 61540/10259907 (6.00 per 1,000) in 2011 [actors: GOV; location contains: United States]
3. "Increase in New York Times articles discussing government interventions in economic crises, such as stimulus packages or unemployment benefits." (source: nyt, predicted: increase)
   Measured: 3/26020 matching articles (0.12 per 1,000) in 2006 vs. 20/16043 (1.25 per 1,000) in 2011 [sections: Business, U.S., Politics]
4. "More GDELT events involving government officials making public statements (01) advocating for increased government responsibility in social welfare." (source: gdelt, predicted: increase)
   Measured: 31108/969651 matching events (32.08 per 1,000) in 2006 vs. 247328/10259907 (24.11 per 1,000) in 2011 [actors: GOV; location contains: United States]
5. "Increase in New York Times opinion pieces and editorials supporting expanded government roles in healthcare, education, and social services." (source: nyt, predicted: increase)
   Measured: 2/16492 matching articles (0.12 per 1,000) in 2006 vs. 1/9196 (0.11 per 1,000) in 2011 [sections: Opinion]

=== Direction B ===
1. "Increase in New York Times headlines emphasizing personal responsibility, self-reliance, and critiques of government dependency." (source: nyt, predicted: increase)
   Measured: 24/27234 matching articles (0.88 per 1,000) in 2006 vs. 2/19036 (0.11 per 1,000) in 2011 [sections: U.S., Opinion, Politics]
2. "More GDELT events coded as government DISAPPROVE (11) or REJECT (12) related to expanding welfare programs or government aid." (source: gdelt, predicted: increase)
   Measured: 24132/969651 matching events (24.89 per 1,000) in 2006 vs. 194342/10259907 (18.94 per 1,000) in 2011 [actors: GOV, LEG; location contains: United States]
3. "Increase in New York Times articles covering political debates or legislation aimed at reducing government welfare programs and promoting work requirements." (source: nyt, predicted: increase)
   Measured: 7/10742 matching articles (0.65 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: Politics, U.S.]
4. "More GDELT events involving protests (14) or public statements (01) by civilians or politicians advocating for reduced government intervention and increased individual responsibility." (source: gdelt, predicted: increase)
   Measured: 43960/969651 matching events (45.34 per 1,000) in 2006 vs. 408913/10259907 (39.86 per 1,000) in 2011 [actors: CVL, GOV, LEG; location contains: United States]
5. "Increase in New York Times opinion pieces criticizing government overreach and promoting free-market solutions and individual accountability." (source: nyt, predicted: increase)
   Measured: 12/16492 matching articles (0.73 per 1,000) in 2006 vs. 1/9196 (0.11 per 1,000) in 2011 [sections: Opinion]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "A", "reasoning": "Indicators for Direction A show a clear increase in government aid events and articles discussing government interventions in economic crises, which strongly supports increased support for government responsibility. In contrast, indicators for Direction B mostly show decreases in relevant articles and events emphasizing personal responsibility, suggesting less support for individual responsibility.", "requested_evidence": []}
```

## 27. F120 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how justifiable people consider abortion, on a scale from 1 (never justifiable) to 10 (always justifiable). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "always justifiable" (attitudes became more accepting of abortion).
Direction B: average answer moved toward "never justifiable" (attitudes became less accepting of abortion).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines or abstracts discussing expansion of abortion rights or access to abortion services.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["abortion rights", "expand abortion access", "reproductive rights", "pro-choice", "abortion clinic"],
      "nyt_sections": ["U.S.", "Opinion", "Health"]
    },
    {
      "indicator": "More GDELT events involving government or legislature actors making public statements supporting abortion rights or reproductive freedom.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["GOV", "LEG"],
      "gdelt_tone": "more_positive",
      "gdelt_goldstein": "more_positive"
    },
    {
      "indicator": "Increase in protests or demonstrations in favor of abortion rights reported in news coverage and GDELT protest event codes.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "More New York Times articles highlighting court rulings or judicial decisions upholding abortion rights or striking down abortion restrictions.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["Supreme Court abortion", "court upholds abortion", "judicial ruling abortion", "abortion law struck down"],
      "nyt_sections": ["U.S.", "Politics"]
    },
    {
      "indicator": "Increase in media coverage and public discourse about contraception and reproductive health as part of broader reproductive rights discussions.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["contraception", "birth control", "reproductive health", "family planning"],
      "nyt_sections": ["Health", "U.S."]
    }
  ],
  "B": [
    {
      "indicator": "Increase in New York Times headlines or abstracts emphasizing abortion restrictions, bans, or legislative efforts to limit abortion access.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["abortion ban", "restrict abortion", "limit abortion access", "pro-life legislation", "abortion restrictions"],
      "nyt_sections": ["U.S.", "Politics"]
    },
    {
      "indicator": "More GDELT events involving religious actors or conservative groups publicly opposing abortion or advocating for pro-life policies.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01", "10", "11"],
      "gdelt_actor_types": ["REL", "CVL"],
      "gdelt_tone": "more_negative",
      "gdelt_goldstein": "more_negative"
    },
    {
      "indicator": "Increase in protests or demonstrations against abortion reported in news and GDELT protest event codes.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "More New York Times articles covering judicial decisions or court rulings that uphold abortion restrictions or allow more restrictive abortion laws.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["court upholds abortion restrictions", "judicial ruling limits abortion", "abortion law upheld"],
      "nyt_sections": ["U.S.", "Politics"]
    },
    {
      "indicator": "Increase in political campaigns or legislative efforts by conservative politicians focusing on pro-life messaging and abortion opposition.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["pro-life campaign", "anti-abortion legislation", "abortion opposition", "conservative abortion stance"],
      "nyt_sections": ["U.S.", "Politics"]
    }
  ]
}
```

## 28. F120 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how justifiable people consider abortion, on a scale from 1 (never justifiable) to 10 (always justifiable). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "always justifiable" (attitudes became more accepting of abortion).
Direction B: average answer moved toward "never justifiable" (attitudes became less accepting of abortion).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines or abstracts discussing expansion of abortion rights or access to abortion services." (source: nyt, predicted: increase)
   Measured: 48/29265 matching articles (1.64 per 1,000) in 2006 vs. 18/21299 (0.85 per 1,000) in 2011 [sections: U.S., Opinion, Health]
2. "More GDELT events involving government or legislature actors making public statements supporting abortion rights or reproductive freedom." (source: gdelt, predicted: increase)
   Measured: 37043/969651 matching events (38.20 per 1,000) in 2006 vs. 312918/10259907 (30.50 per 1,000) in 2011 [actors: GOV, LEG]; avg tone 5.73 vs. 5.63; avg Goldstein 0.02 vs. 0.03
3. "Increase in protests or demonstrations in favor of abortion rights reported in news coverage and GDELT protest event codes." (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL]; avg tone 5.51 vs. 6.09
4. "More New York Times articles highlighting court rulings or judicial decisions upholding abortion rights or striking down abortion restrictions." (source: nyt, predicted: increase)
   Measured: 0/10742 matching articles (0.00 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Politics]
5. "Increase in media coverage and public discourse about contraception and reproductive health as part of broader reproductive rights discussions." (source: nyt, predicted: increase)
   Measured: 22/12773 matching articles (1.72 per 1,000) in 2006 vs. 25/12103 (2.07 per 1,000) in 2011 [sections: Health, U.S.]

=== Direction B ===
1. "Increase in New York Times headlines or abstracts emphasizing abortion restrictions, bans, or legislative efforts to limit abortion access." (source: nyt, predicted: increase)
   Measured: 19/10742 matching articles (1.77 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Politics]
2. "More GDELT events involving religious actors or conservative groups publicly opposing abortion or advocating for pro-life policies." (source: gdelt, predicted: increase)
   Measured: 9356/969651 matching events (9.65 per 1,000) in 2006 vs. 126707/10259907 (12.35 per 1,000) in 2011 [actors: REL, CVL]; avg tone 5.51 vs. 5.57; avg Goldstein -0.88 vs. -0.83
3. "Increase in protests or demonstrations against abortion reported in news and GDELT protest event codes." (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL]; avg tone 5.51 vs. 6.09
4. "More New York Times articles covering judicial decisions or court rulings that uphold abortion restrictions or allow more restrictive abortion laws." (source: nyt, predicted: increase)
   Measured: 0/10742 matching articles (0.00 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Politics]
5. "Increase in political campaigns or legislative efforts by conservative politicians focusing on pro-life messaging and abortion opposition." (source: nyt, predicted: increase)
   Measured: 0/10742 matching articles (0.00 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Politics]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "B", "reasoning": "The evidence shows a decrease in media coverage and government support for abortion rights (Direction A indicators mostly decreased or stayed flat), while indicators for opposition to abortion (Direction B) such as GDELT events involving religious and conservative actors increased notably. Protests against abortion also increased, supporting Direction B. The lack of increase in supportive court rulings or political campaigns for abortion rights further weakens Direction A.", "requested_evidence": []}
```

## 29. F123 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how justifiable people consider suicide, on a scale from 1 (never justifiable) to 10 (always justifiable). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "always justifiable" (attitudes became more accepting of suicide).
Direction B: average answer moved toward "never justifiable" (attitudes became less accepting of suicide).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times articles discussing suicide as a personal right or a form of autonomy.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["suicide as a right", "right to die", "autonomy in suicide", "assisted suicide", "end of life choice"],
      "nyt_sections": ["U.S.", "Health", "Opinion"]
    },
    {
      "indicator": "More GDELT events involving public figures or organizations advocating for legalization or acceptance of assisted suicide or euthanasia.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["02", "10"],
      "gdelt_actor_types": ["GOV", "REL", "MED"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in New York Times headlines or abstracts reporting on rising suicide rates or mental health crises without strong condemnation.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["suicide rates rise", "mental health crisis", "suicide epidemic", "suicide statistics"],
      "nyt_sections": ["U.S.", "Health"]
    },
    {
      "indicator": "More GDELT protest events or public demonstrations supporting suicide prevention reforms that emphasize personal choice rather than prohibition.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL", "MED"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in New York Times opinion pieces or editorials framing suicide as a complex social issue deserving empathy rather than moral judgment.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["suicide empathy", "understanding suicide", "mental health and suicide", "suicide stigma reduction"],
      "nyt_sections": ["Opinion", "Health"]
    }
  ],
  "B": [
    {
      "indicator": "Increase in New York Times articles condemning suicide and emphasizing its preventability and moral wrongness.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["suicide prevention", "suicide is wrong", "moral condemnation of suicide", "suicide crisis response"],
      "nyt_sections": ["U.S.", "Opinion", "Health"]
    },
    {
      "indicator": "More GDELT events involving government or religious actors issuing statements or demands against suicide or assisted suicide.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01", "10"],
      "gdelt_actor_types": ["GOV", "REL"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in New York Times coverage of suicide prevention programs, campaigns, and legislation aimed at reducing suicide rates.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["suicide prevention program", "anti-suicide campaign", "suicide reduction law", "mental health intervention"],
      "nyt_sections": ["Health", "U.S."]
    },
    {
      "indicator": "More GDELT events reporting police or emergency responses to suicide attempts, emphasizing intervention and rescue.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["09", "11"],
      "gdelt_actor_types": ["COP", "GOV"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in New York Times opinion pieces advocating for stronger social and legal measures to discourage suicide.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["discourage suicide", "anti-suicide laws", "suicide prevention advocacy", "moral opposition to suicide"],
      "nyt_sections": ["Opinion", "U.S."]
    }
  ]
}
```

## 30. F123 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how justifiable people consider suicide, on a scale from 1 (never justifiable) to 10 (always justifiable). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "always justifiable" (attitudes became more accepting of suicide).
Direction B: average answer moved toward "never justifiable" (attitudes became less accepting of suicide).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times articles discussing suicide as a personal right or a form of autonomy." (source: nyt, predicted: increase)
   Measured: 12/29265 matching articles (0.41 per 1,000) in 2006 vs. 5/21299 (0.23 per 1,000) in 2011 [sections: U.S., Health, Opinion]
2. "More GDELT events involving public figures or organizations advocating for legalization or acceptance of assisted suicide or euthanasia." (source: gdelt, predicted: increase)
   Measured: 26619/969651 matching events (27.45 per 1,000) in 2006 vs. 223845/10259907 (21.82 per 1,000) in 2011 [actors: GOV, REL, MED]; avg tone 6.62 vs. 5.82
3. "Increase in New York Times headlines or abstracts reporting on rising suicide rates or mental health crises without strong condemnation." (source: nyt, predicted: increase)
   Measured: 0/12773 matching articles (0.00 per 1,000) in 2006 vs. 0/12103 (0.00 per 1,000) in 2011 [sections: U.S., Health]
4. "More GDELT protest events or public demonstrations supporting suicide prevention reforms that emphasize personal choice rather than prohibition." (source: gdelt, predicted: increase)
   Measured: 412/969651 matching events (0.42 per 1,000) in 2006 vs. 6177/10259907 (0.60 per 1,000) in 2011 [actors: CVL, MED]; avg tone 5.37 vs. 5.77
5. "Increase in New York Times opinion pieces or editorials framing suicide as a complex social issue deserving empathy rather than moral judgment." (source: nyt, predicted: increase)
   Measured: 0/18523 matching articles (0.00 per 1,000) in 2006 vs. 0/11459 (0.00 per 1,000) in 2011 [sections: Opinion, Health]

=== Direction B ===
1. "Increase in New York Times articles condemning suicide and emphasizing its preventability and moral wrongness." (source: nyt, predicted: increase)
   Measured: 0/29265 matching articles (0.00 per 1,000) in 2006 vs. 0/21299 (0.00 per 1,000) in 2011 [sections: U.S., Opinion, Health]
2. "More GDELT events involving government or religious actors issuing statements or demands against suicide or assisted suicide." (source: gdelt, predicted: increase)
   Measured: 34291/969651 matching events (35.36 per 1,000) in 2006 vs. 272931/10259907 (26.60 per 1,000) in 2011 [actors: GOV, REL]; avg tone 5.71 vs. 5.61
3. "Increase in New York Times coverage of suicide prevention programs, campaigns, and legislation aimed at reducing suicide rates." (source: nyt, predicted: increase)
   Measured: 0/12773 matching articles (0.00 per 1,000) in 2006 vs. 0/12103 (0.00 per 1,000) in 2011 [sections: Health, U.S.]
4. "More GDELT events reporting police or emergency responses to suicide attempts, emphasizing intervention and rescue." (source: gdelt, predicted: increase)
   Measured: 21890/969651 matching events (22.58 per 1,000) in 2006 vs. 227429/10259907 (22.17 per 1,000) in 2011 [actors: COP, GOV]; avg tone 5.09 vs. 4.39
5. "Increase in New York Times opinion pieces advocating for stronger social and legal measures to discourage suicide." (source: nyt, predicted: increase)
   Measured: 0/27234 matching articles (0.00 per 1,000) in 2006 vs. 0/19036 (0.00 per 1,000) in 2011 [sections: Opinion, U.S.]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "B", "reasoning": "The evidence shows a decrease in indicators supporting acceptance of suicide (Direction A) and a decrease or no change in indicators supporting acceptance, while indicators opposing suicide (Direction B) remain higher in volume and tone, despite some decreases. Notably, government and religious actors' statements against suicide remain substantial, and police/emergency responses emphasizing intervention are steady, suggesting attitudes became less accepting of suicide.", "requested_evidence": []}
```

## 31. C006 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: people's satisfaction with their household's financial situation, on a scale from 1 (dissatisfied) to 10 (satisfied). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "satisfied" (people became more satisfied with their household finances).
Direction B: average answer moved toward "dissatisfied" (people became less satisfied with their household finances).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in NYT headlines about rising household incomes or improved personal finances",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["household income increase", "personal finance improvement", "financial satisfaction", "economic recovery"],
      "nyt_sections": ["Business", "U.S."]
    },
    {
      "indicator": "More GDELT events coded as '06 ENGAGE IN MATERIAL COOPERATION' involving business and government actors indicating economic stimulus or financial aid programs",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["06"],
      "gdelt_actor_types": ["GOV", "BUS"],
      "gdelt_tone": "more_positive",
      "gdelt_goldstein": "more_positive"
    },
    {
      "indicator": "Increase in NYT articles discussing stock market gains or housing market recovery",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["stock market rise", "housing market recovery", "real estate prices up"],
      "nyt_sections": ["Business"]
    },
    {
      "indicator": "More GDELT events with '01 MAKE PUBLIC STATEMENT' by government officials about economic growth or financial stability",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["GOV"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in NYT coverage of consumer confidence indices rising or positive economic outlook reports",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["consumer confidence up", "economic outlook positive", "financial optimism"],
      "nyt_sections": ["Business", "U.S."]
    }
  ],
  "B": [
    {
      "indicator": "Increase in NYT headlines about rising unemployment, foreclosures, or household debt problems",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["unemployment rise", "foreclosure crisis", "household debt", "financial distress"],
      "nyt_sections": ["U.S.", "Business"]
    },
    {
      "indicator": "More GDELT events coded as '14 PROTEST' or '11 DISAPPROVE' involving civilians or labor groups protesting economic conditions",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14", "11"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_tone": "more_negative",
      "gdelt_goldstein": "more_negative"
    },
    {
      "indicator": "Increase in NYT articles discussing stock market crashes, housing market collapse, or recession fears",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["stock market crash", "housing market collapse", "recession fears", "economic downturn"],
      "nyt_sections": ["Business"]
    },
    {
      "indicator": "More GDELT events with '10 DEMAND' or '12 REJECT' codes involving government or business actors rejecting economic recovery plans or facing criticism",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["10", "12"],
      "gdelt_actor_types": ["GOV", "BUS"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in NYT coverage of consumer confidence indices falling or reports of financial insecurity",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["consumer confidence down", "financial insecurity", "economic anxiety"],
      "nyt_sections": ["Business", "U.S."]
    }
  ]
}
```

## 32. C006 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: people's satisfaction with their household's financial situation, on a scale from 1 (dissatisfied) to 10 (satisfied). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "satisfied" (people became more satisfied with their household finances).
Direction B: average answer moved toward "dissatisfied" (people became less satisfied with their household finances).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in NYT headlines about rising household incomes or improved personal finances" (source: nyt, predicted: increase)
   Measured: 0/26020 matching articles (0.00 per 1,000) in 2006 vs. 22/16043 (1.37 per 1,000) in 2011 [sections: Business, U.S.]
2. "More GDELT events coded as '06 ENGAGE IN MATERIAL COOPERATION' involving business and government actors indicating economic stimulus or financial aid programs" (source: gdelt, predicted: increase)
   Measured: 5879/969651 matching events (6.06 per 1,000) in 2006 vs. 56431/10259907 (5.50 per 1,000) in 2011 [actors: GOV, BUS]; avg tone 6.49 vs. 6.14; avg Goldstein 6.16 vs. 6.16
3. "Increase in NYT articles discussing stock market gains or housing market recovery" (source: nyt, predicted: increase)
   Measured: 0/15279 matching articles (0.00 per 1,000) in 2006 vs. 0/6218 (0.00 per 1,000) in 2011 [sections: Business]
4. "More GDELT events with '01 MAKE PUBLIC STATEMENT' by government officials about economic growth or financial stability" (source: gdelt, predicted: increase)
   Measured: 31108/969651 matching events (32.08 per 1,000) in 2006 vs. 247328/10259907 (24.11 per 1,000) in 2011 [actors: GOV]; avg tone 5.70 vs. 5.61
5. "Increase in NYT coverage of consumer confidence indices rising or positive economic outlook reports" (source: nyt, predicted: increase)
   Measured: 0/26020 matching articles (0.00 per 1,000) in 2006 vs. 0/16043 (0.00 per 1,000) in 2011 [sections: Business, U.S.]

=== Direction B ===
1. "Increase in NYT headlines about rising unemployment, foreclosures, or household debt problems" (source: nyt, predicted: increase)
   Measured: 3/26020 matching articles (0.12 per 1,000) in 2006 vs. 3/16043 (0.19 per 1,000) in 2011 [sections: U.S., Business]
2. "More GDELT events coded as '14 PROTEST' or '11 DISAPPROVE' involving civilians or labor groups protesting economic conditions" (source: gdelt, predicted: increase)
   Measured: 2933/969651 matching events (3.02 per 1,000) in 2006 vs. 34879/10259907 (3.40 per 1,000) in 2011 [actors: CVL]; avg tone 5.29 vs. 5.41; avg Goldstein -2.43 vs. -2.49
3. "Increase in NYT articles discussing stock market crashes, housing market collapse, or recession fears" (source: nyt, predicted: increase)
   Measured: 1/15279 matching articles (0.07 per 1,000) in 2006 vs. 5/6218 (0.80 per 1,000) in 2011 [sections: Business]
4. "More GDELT events with '10 DEMAND' or '12 REJECT' codes involving government or business actors rejecting economic recovery plans or facing criticism" (source: gdelt, predicted: increase)
   Measured: 11781/969651 matching events (12.15 per 1,000) in 2006 vs. 99350/10259907 (9.68 per 1,000) in 2011 [actors: GOV, BUS]; avg tone 5.89 vs. 5.75
5. "Increase in NYT coverage of consumer confidence indices falling or reports of financial insecurity" (source: nyt, predicted: increase)
   Measured: 1/26020 matching articles (0.04 per 1,000) in 2006 vs. 2/16043 (0.12 per 1,000) in 2011 [sections: Business, U.S.]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "B", "reasoning": "The evidence shows a clear increase in negative economic indicators and public dissatisfaction between 2006 and 2011, including more NYT headlines about unemployment and debt problems, more protests and disapproval events, and increased discussion of stock market crashes and recession fears. Positive indicators for improved satisfaction are weak or absent, with no increase in articles about rising incomes or consumer confidence and a decrease in government public statements about economic growth.", "requested_evidence": []}
```

## 33. E234 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people think the world is better off or worse off because of science and technology, on a scale from 1 (a lot worse off) to 10 (a lot better off). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "a lot better off" (people became more positive about the overall effect of science and technology on the world).
Direction B: average answer moved toward "a lot worse off" (people became more negative about the overall effect of science and technology on the world).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines or abstracts highlighting breakthroughs and benefits of science and technology improving health, environment, or economy.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["scientific breakthrough", "technology innovation", "medical advance", "clean energy", "economic growth technology"],
      "nyt_sections": ["Science", "Technology", "Health", "Business"]
    },
    {
      "indicator": "Increase in GDELT events coded as 06 ENGAGE IN MATERIAL COOPERATION involving scientific institutions or technology companies promoting new technologies or collaborations.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["06"],
      "gdelt_actor_types": ["BUS", "EDU", "GOV"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in New York Times opinion pieces or editorials expressing optimism about the role of science and technology in solving global problems.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["science optimism", "technology benefits", "future of innovation", "technology solving problems"],
      "nyt_sections": ["Opinion", "Science", "Technology"]
    },
    {
      "indicator": "Increase in GDELT events coded as 01 MAKE PUBLIC STATEMENT by government or scientific leaders praising science and technology progress.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["GOV", "EDU"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in New York Times coverage of successful technology product launches or adoption stories that emphasize positive societal impact.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["technology launch", "new device", "tech adoption", "innovation success"],
      "nyt_sections": ["Business", "Technology"]
    }
  ],
  "B": [
    {
      "indicator": "Increase in New York Times headlines or abstracts focusing on negative consequences or risks of science and technology, such as privacy breaches, environmental damage, or ethical concerns.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["technology risks", "privacy breach", "environmental damage technology", "ethical concerns science", "technology backlash"],
      "nyt_sections": ["Science", "Technology", "Opinion"]
    },
    {
      "indicator": "Increase in GDELT events coded as 14 PROTEST or 11 DISAPPROVE involving civilians or activists opposing technological developments or scientific projects.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14", "11"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in New York Times opinion pieces or editorials expressing skepticism or criticism about the impact of science and technology on society.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["technology skepticism", "science criticism", "technology harms", "science ethical issues"],
      "nyt_sections": ["Opinion", "Science", "Technology"]
    },
    {
      "indicator": "Increase in GDELT events coded as 12 REJECT or 10 DEMAND involving government or public actors calling for regulation or restriction of certain technologies.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["12", "10"],
      "gdelt_actor_types": ["GOV", "CVL"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in New York Times coverage of technology failures, accidents, or scandals that highlight harmful effects or unintended consequences.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["technology failure", "science scandal", "accident technology", "technology harm"],
      "nyt_sections": ["Science", "Technology", "U.S."]
    }
  ]
}
```

## 34. E234 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people think the world is better off or worse off because of science and technology, on a scale from 1 (a lot worse off) to 10 (a lot better off). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "a lot better off" (people became more positive about the overall effect of science and technology on the world).
Direction B: average answer moved toward "a lot worse off" (people became more negative about the overall effect of science and technology on the world).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines or abstracts highlighting breakthroughs and benefits of science and technology improving health, environment, or economy." (source: nyt, predicted: increase)
   Measured: 9/19049 matching articles (0.47 per 1,000) in 2006 vs. 7/12528 (0.56 per 1,000) in 2011 [sections: Science, Technology, Health, Business]
2. "Increase in GDELT events coded as 06 ENGAGE IN MATERIAL COOPERATION involving scientific institutions or technology companies promoting new technologies or collaborations." (source: gdelt, predicted: increase)
   Measured: 6412/969651 matching events (6.61 per 1,000) in 2006 vs. 69048/10259907 (6.73 per 1,000) in 2011 [actors: BUS, EDU, GOV]; avg tone 6.53 vs. 6.34
3. "Increase in New York Times opinion pieces or editorials expressing optimism about the role of science and technology in solving global problems." (source: nyt, predicted: increase)
   Measured: 0/19780 matching articles (0.00 per 1,000) in 2006 vs. 0/15118 (0.00 per 1,000) in 2011 [sections: Opinion, Science, Technology]
4. "Increase in GDELT events coded as 01 MAKE PUBLIC STATEMENT by government or scientific leaders praising science and technology progress." (source: gdelt, predicted: increase)
   Measured: 36105/969651 matching events (37.24 per 1,000) in 2006 vs. 341620/10259907 (33.30 per 1,000) in 2011 [actors: GOV, EDU]; avg tone 5.74 vs. 5.78
5. "Increase in New York Times coverage of successful technology product launches or adoption stories that emphasize positive societal impact." (source: nyt, predicted: increase)
   Measured: 3/16175 matching articles (0.19 per 1,000) in 2006 vs. 5/8117 (0.62 per 1,000) in 2011 [sections: Business, Technology]

=== Direction B ===
1. "Increase in New York Times headlines or abstracts focusing on negative consequences or risks of science and technology, such as privacy breaches, environmental damage, or ethical concerns." (source: nyt, predicted: increase)
   Measured: 0/19780 matching articles (0.00 per 1,000) in 2006 vs. 1/15118 (0.07 per 1,000) in 2011 [sections: Science, Technology, Opinion]
2. "Increase in GDELT events coded as 14 PROTEST or 11 DISAPPROVE involving civilians or activists opposing technological developments or scientific projects." (source: gdelt, predicted: increase)
   Measured: 2933/969651 matching events (3.02 per 1,000) in 2006 vs. 34879/10259907 (3.40 per 1,000) in 2011 [actors: CVL]; avg tone 5.29 vs. 5.41
3. "Increase in New York Times opinion pieces or editorials expressing skepticism or criticism about the impact of science and technology on society." (source: nyt, predicted: increase)
   Measured: 0/19780 matching articles (0.00 per 1,000) in 2006 vs. 0/15118 (0.00 per 1,000) in 2011 [sections: Opinion, Science, Technology]
4. "Increase in GDELT events coded as 12 REJECT or 10 DEMAND involving government or public actors calling for regulation or restriction of certain technologies." (source: gdelt, predicted: increase)
   Measured: 10799/969651 matching events (11.14 per 1,000) in 2006 vs. 97499/10259907 (9.50 per 1,000) in 2011 [actors: GOV, CVL]; avg tone 5.81 vs. 5.64
5. "Increase in New York Times coverage of technology failures, accidents, or scandals that highlight harmful effects or unintended consequences." (source: nyt, predicted: increase)
   Measured: 0/14030 matching articles (0.00 per 1,000) in 2006 vs. 0/15762 (0.00 per 1,000) in 2011 [sections: Science, Technology, U.S.]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "A", "reasoning": "The indicators for Direction A mostly show slight increases or stable positive signals in media coverage and events highlighting benefits and cooperation in science and technology, while Direction B indicators show minimal or no increase in negative coverage or protest events. The positive indicators, though modest, are more consistent and stronger than the negative ones, which are mostly zero or declining.", "requested_evidence": []}
```

## 35. E041 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people believe wealth can only be accumulated at others' expense, or that wealth can grow so there is enough for everyone, on a scale from 1 (people can only get rich at the expense of others) to 10 (wealth can grow so there's enough for everyone). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "wealth can grow so there's enough for everyone" (people became more optimistic that wealth creation benefits everyone).
Direction B: average answer moved toward "people can only get rich at the expense of others" (people became more convinced that wealth comes at others' expense).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines emphasizing economic growth benefiting all or rising tide lifting all boats",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["economic growth benefits all", "rising tide lifts all boats", "shared prosperity", "wealth creation for everyone"],
      "nyt_sections": ["Business", "Economy", "Opinion"]
    },
    {
      "indicator": "More GDELT events involving business cooperation or partnerships aimed at expanding markets or innovation",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["03", "05", "06"],
      "gdelt_actor_types": ["BUS", "GOV"],
      "gdelt_tone": "more_positive",
      "gdelt_goldstein": "more_positive"
    },
    {
      "indicator": "Increase in NYT articles discussing policies or initiatives promoting entrepreneurship and wealth creation opportunities for all",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["entrepreneurship", "small business growth", "wealth creation opportunities", "inclusive economy"],
      "nyt_sections": ["Business", "U.S."]
    },
    {
      "indicator": "More GDELT events reporting government or business announcements about economic stimulus, job creation, or innovation funding",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01", "03", "06"],
      "gdelt_actor_types": ["GOV", "BUS"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Decrease in NYT headlines focusing on zero-sum economic conflict or wealth redistribution as a primary theme",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["wealth redistribution", "zero-sum", "economic conflict", "rich get richer at expense"],
      "nyt_sections": ["Opinion", "Business"]
    }
  ],
  "B": [
    {
      "indicator": "Increase in NYT headlines emphasizing economic inequality, wealth concentration, and the rich exploiting others",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["economic inequality", "wealth concentration", "rich exploit", "wealth at expense of others", "income disparity"],
      "nyt_sections": ["Opinion", "Business", "U.S."]
    },
    {
      "indicator": "More GDELT protest events related to economic injustice, wealth inequality, or anti-corporate demonstrations",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_tone": "more_negative",
      "gdelt_goldstein": "more_negative"
    },
    {
      "indicator": "Increase in NYT articles discussing failures of trickle-down economics or critiques of capitalism as inherently exploitative",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["trickle-down economics failure", "capitalism exploitative", "wealth extraction", "economic injustice"],
      "nyt_sections": ["Opinion", "Business"]
    },
    {
      "indicator": "More GDELT events involving government investigations or demands related to corporate malfeasance, fraud, or economic abuse",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["09", "10"],
      "gdelt_actor_types": ["GOV"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Decrease in NYT headlines about economic cooperation, shared prosperity, or wealth creation benefiting all",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["shared prosperity", "economic cooperation", "wealth creation for everyone", "inclusive growth"],
      "nyt_sections": ["Business", "Economy"]
    }
  ]
}
```

## 36. E041 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people believe wealth can only be accumulated at others' expense, or that wealth can grow so there is enough for everyone, on a scale from 1 (people can only get rich at the expense of others) to 10 (wealth can grow so there's enough for everyone). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "wealth can grow so there's enough for everyone" (people became more optimistic that wealth creation benefits everyone).
Direction B: average answer moved toward "people can only get rich at the expense of others" (people became more convinced that wealth comes at others' expense).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines emphasizing economic growth benefiting all or rising tide lifting all boats" (source: nyt, predicted: increase)
   Measured: 0/31771 matching articles (0.00 per 1,000) in 2006 vs. 0/15414 (0.00 per 1,000) in 2011 [sections: Business, Economy, Opinion]
2. "More GDELT events involving business cooperation or partnerships aimed at expanding markets or innovation" (source: gdelt, predicted: increase)
   Measured: 51647/969651 matching events (53.26 per 1,000) in 2006 vs. 456099/10259907 (44.45 per 1,000) in 2011 [actors: BUS, GOV]; avg tone 6.68 vs. 6.66; avg Goldstein 4.40 vs. 4.35
3. "Increase in NYT articles discussing policies or initiatives promoting entrepreneurship and wealth creation opportunities for all" (source: nyt, predicted: increase)
   Measured: 4/26020 matching articles (0.15 per 1,000) in 2006 vs. 2/16043 (0.12 per 1,000) in 2011 [sections: Business, U.S.]
4. "More GDELT events reporting government or business announcements about economic stimulus, job creation, or innovation funding" (source: gdelt, predicted: increase)
   Measured: 76150/969651 matching events (78.53 per 1,000) in 2006 vs. 613290/10259907 (59.78 per 1,000) in 2011 [actors: GOV, BUS]; avg tone 6.27 vs. 5.97
5. "Decrease in NYT headlines focusing on zero-sum economic conflict or wealth redistribution as a primary theme" (source: nyt, predicted: decrease)
   Measured: 4/31771 matching articles (0.13 per 1,000) in 2006 vs. 0/15414 (0.00 per 1,000) in 2011 [sections: Opinion, Business]

=== Direction B ===
1. "Increase in NYT headlines emphasizing economic inequality, wealth concentration, and the rich exploiting others" (source: nyt, predicted: increase)
   Measured: 1/42512 matching articles (0.02 per 1,000) in 2006 vs. 3/25239 (0.12 per 1,000) in 2011 [sections: Opinion, Business, U.S.]
2. "More GDELT protest events related to economic injustice, wealth inequality, or anti-corporate demonstrations" (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL]; avg tone 5.51 vs. 6.09; avg Goldstein -6.56 vs. -6.57
3. "Increase in NYT articles discussing failures of trickle-down economics or critiques of capitalism as inherently exploitative" (source: nyt, predicted: increase)
   Measured: 0/31771 matching articles (0.00 per 1,000) in 2006 vs. 0/15414 (0.00 per 1,000) in 2011 [sections: Opinion, Business]
4. "More GDELT events involving government investigations or demands related to corporate malfeasance, fraud, or economic abuse" (source: gdelt, predicted: increase)
   Measured: 7240/969651 matching events (7.47 per 1,000) in 2006 vs. 61906/10259907 (6.03 per 1,000) in 2011 [actors: GOV]; avg tone 5.34 vs. 4.97
5. "Decrease in NYT headlines about economic cooperation, shared prosperity, or wealth creation benefiting all" (source: nyt, predicted: decrease)
   Measured: 1/15279 matching articles (0.07 per 1,000) in 2006 vs. 1/6218 (0.16 per 1,000) in 2011 [sections: Business, Economy]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "B", "reasoning": "The evidence shows a clear increase in indicators related to economic inequality and exploitation, such as more NYT headlines on wealth concentration and anti-corporate protests, despite some decreases in positive economic cooperation events. The rise in protest events and critical headlines, along with a slight increase in negative tone in protests, supports a shift toward the belief that wealth is accumulated at others' expense.", "requested_evidence": []}
```

## 37. E233 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people see women having the same rights as men as an essential characteristic of democracy, on a scale from 0 (against democracy) to 10 (an essential characteristic of democracy). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "an essential characteristic of democracy" (people placed more importance on equal rights for women as part of democracy).
Direction B: average answer moved toward lower importance (people placed less importance on equal rights for women as part of democracy).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines or abstracts emphasizing women's equal rights as fundamental to democracy.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["women's rights", "equal rights", "democracy", "gender equality", "women's equality"],
      "nyt_sections": ["U.S.", "Opinion", "Politics"]
    },
    {
      "indicator": "More GDELT events involving government or legislative actors making public statements supporting women's equal rights as essential to democracy.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["GOV", "LEG"],
      "gdelt_tone": "more_positive",
      "gdelt_goldstein": "more_positive"
    },
    {
      "indicator": "Increase in protests or public demonstrations advocating for women's equal rights and gender equality in democratic participation.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "More New York Times articles highlighting legislative or judicial advances in women's rights, such as equal pay laws or anti-discrimination rulings.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["equal pay", "anti-discrimination", "women's rights legislation", "gender equality law", "court ruling on women's rights"],
      "nyt_sections": ["U.S.", "Politics", "Business"]
    },
    {
      "indicator": "Increase in GDELT events where education or media actors promote gender equality and women's rights as core democratic values.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01", "03"],
      "gdelt_actor_types": ["EDU", "MED"],
      "gdelt_tone": "more_positive"
    }
  ],
  "B": [
    {
      "indicator": "Decrease in New York Times headlines or abstracts emphasizing women's equal rights as fundamental to democracy.",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["women's rights", "equal rights", "democracy", "gender equality", "women's equality"],
      "nyt_sections": ["U.S.", "Opinion", "Politics"]
    },
    {
      "indicator": "Fewer GDELT events involving government or legislative actors making public statements supporting women's equal rights as essential to democracy.",
      "source": "gdelt",
      "expected_change": "decrease",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["GOV", "LEG"],
      "gdelt_tone": "more_positive",
      "gdelt_goldstein": "more_positive"
    },
    {
      "indicator": "Decrease in protests or public demonstrations advocating for women's equal rights and gender equality in democratic participation.",
      "source": "gdelt",
      "expected_change": "decrease",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Fewer New York Times articles highlighting legislative or judicial advances in women's rights, such as equal pay laws or anti-discrimination rulings.",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["equal pay", "anti-discrimination", "women's rights legislation", "gender equality law", "court ruling on women's rights"],
      "nyt_sections": ["U.S.", "Politics", "Business"]
    },
    {
      "indicator": "Decrease in GDELT events where education or media actors promote gender equality and women's rights as core democratic values.",
      "source": "gdelt",
      "expected_change": "decrease",
      "gdelt_event_codes": ["01", "03"],
      "gdelt_actor_types": ["EDU", "MED"],
      "gdelt_tone": "more_positive"
    }
  ]
}
```

## 38. E233 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people see women having the same rights as men as an essential characteristic of democracy, on a scale from 0 (against democracy) to 10 (an essential characteristic of democracy). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "an essential characteristic of democracy" (people placed more importance on equal rights for women as part of democracy).
Direction B: average answer moved toward lower importance (people placed less importance on equal rights for women as part of democracy).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines or abstracts emphasizing women's equal rights as fundamental to democracy." (source: nyt, predicted: increase)
   Measured: 284/27234 matching articles (10.43 per 1,000) in 2006 vs. 103/19036 (5.41 per 1,000) in 2011 [sections: U.S., Opinion, Politics]
2. "More GDELT events involving government or legislative actors making public statements supporting women's equal rights as essential to democracy." (source: gdelt, predicted: increase)
   Measured: 37043/969651 matching events (38.20 per 1,000) in 2006 vs. 312918/10259907 (30.50 per 1,000) in 2011 [actors: GOV, LEG]; avg tone 5.73 vs. 5.63; avg Goldstein 0.02 vs. 0.03
3. "Increase in protests or public demonstrations advocating for women's equal rights and gender equality in democratic participation." (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL]; avg tone 5.51 vs. 6.09
4. "More New York Times articles highlighting legislative or judicial advances in women's rights, such as equal pay laws or anti-discrimination rulings." (source: nyt, predicted: increase)
   Measured: 2/26020 matching articles (0.08 per 1,000) in 2006 vs. 0/16043 (0.00 per 1,000) in 2011 [sections: U.S., Politics, Business]
5. "Increase in GDELT events where education or media actors promote gender equality and women's rights as core democratic values." (source: gdelt, predicted: increase)
   Measured: 29883/969651 matching events (30.82 per 1,000) in 2006 vs. 348558/10259907 (33.97 per 1,000) in 2011 [actors: EDU, MED]; avg tone 6.37 vs. 6.02

=== Direction B ===
1. "Decrease in New York Times headlines or abstracts emphasizing women's equal rights as fundamental to democracy." (source: nyt, predicted: decrease)
   Measured: 284/27234 matching articles (10.43 per 1,000) in 2006 vs. 103/19036 (5.41 per 1,000) in 2011 [sections: U.S., Opinion, Politics]
2. "Fewer GDELT events involving government or legislative actors making public statements supporting women's equal rights as essential to democracy." (source: gdelt, predicted: decrease)
   Measured: 37043/969651 matching events (38.20 per 1,000) in 2006 vs. 312918/10259907 (30.50 per 1,000) in 2011 [actors: GOV, LEG]; avg tone 5.73 vs. 5.63; avg Goldstein 0.02 vs. 0.03
3. "Decrease in protests or public demonstrations advocating for women's equal rights and gender equality in democratic participation." (source: gdelt, predicted: decrease)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL]; avg tone 5.51 vs. 6.09
4. "Fewer New York Times articles highlighting legislative or judicial advances in women's rights, such as equal pay laws or anti-discrimination rulings." (source: nyt, predicted: decrease)
   Measured: 2/26020 matching articles (0.08 per 1,000) in 2006 vs. 0/16043 (0.00 per 1,000) in 2011 [sections: U.S., Politics, Business]
5. "Decrease in GDELT events where education or media actors promote gender equality and women's rights as core democratic values." (source: gdelt, predicted: decrease)
   Measured: 29883/969651 matching events (30.82 per 1,000) in 2006 vs. 348558/10259907 (33.97 per 1,000) in 2011 [actors: EDU, MED]; avg tone 6.37 vs. 6.02

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision":"B","reasoning":"Most indicators show a decrease in emphasis on women's equal rights as essential to democracy between 2006 and 2011, including fewer New York Times articles and fewer government or legislative statements supporting this view. Although protests advocating for women's rights increased slightly, this is outweighed by the overall decline in media and official support indicators.","requested_evidence":[]}
```

## 39. D059 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people agree that men make better political leaders than women do, coded 1 (agree strongly) to 4 (strongly disagree). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "strongly disagree" (people became less likely to think men make better political leaders than women).
Direction B: average answer moved toward "agree strongly" (people became more likely to think men make better political leaders than women).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines or abstracts highlighting women winning major political offices or breaking political leadership barriers.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["woman elected", "female mayor", "woman governor", "female senator", "woman political leader"],
      "nyt_sections": ["U.S.", "Politics"]
    },
    {
      "indicator": "More New York Times opinion pieces or editorials advocating for gender equality in political leadership roles.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["gender equality", "women leaders", "female political leadership", "breaking glass ceiling"],
      "nyt_sections": ["Opinion", "U.S."]
    },
    {
      "indicator": "Increase in GDELT events coded as '01 MAKE PUBLIC STATEMENT' or '02 APPEAL' by government or civil society actors promoting women's political participation or leadership.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01", "02"],
      "gdelt_actor_types": ["GOV", "CVL"],
      "gdelt_location_contains": "United States"
    },
    {
      "indicator": "Increase in GDELT events involving protests or demonstrations supporting women's rights or opposing gender discrimination in politics.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_location_contains": "United States"
    },
    {
      "indicator": "More New York Times coverage of high-profile female political candidates or leaders receiving positive media tone.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["female candidate", "woman senator", "woman governor", "female political leader"],
      "nyt_sections": ["Politics", "U.S."]
    }
  ],
  "B": [
    {
      "indicator": "Increase in New York Times headlines or abstracts emphasizing male political leaders' successes or dominance in politics.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["male political leader", "man elected", "male governor", "male senator", "men in politics"],
      "nyt_sections": ["U.S.", "Politics"]
    },
    {
      "indicator": "More New York Times opinion pieces or editorials expressing skepticism about women's effectiveness in political leadership or advocating traditional gender roles.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["women in politics skepticism", "men better leaders", "traditional gender roles", "female leadership doubts"],
      "nyt_sections": ["Opinion", "U.S."]
    },
    {
      "indicator": "Increase in GDELT events coded as '11 DISAPPROVE' or '12 REJECT' involving actors opposing women's political participation or leadership.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["11", "12"],
      "gdelt_actor_types": ["GOV", "REL"],
      "gdelt_location_contains": "United States"
    },
    {
      "indicator": "Increase in GDELT events involving protests or demonstrations opposing women's political rights or leadership roles.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_location_contains": "United States"
    },
    {
      "indicator": "Decrease in New York Times coverage of female political leaders or candidates, or less positive media tone about women in politics.",
      "source": "nyt",
      "expected_change": "decrease",
      "nyt_keywords": ["female candidate", "woman senator", "woman governor", "female political leader"],
      "nyt_sections": ["Politics", "U.S."]
    }
  ]
}
```

## 40. D059 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people agree that men make better political leaders than women do, coded 1 (agree strongly) to 4 (strongly disagree). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "strongly disagree" (people became less likely to think men make better political leaders than women).
Direction B: average answer moved toward "agree strongly" (people became more likely to think men make better political leaders than women).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines or abstracts highlighting women winning major political offices or breaking political leadership barriers." (source: nyt, predicted: increase)
   Measured: 2/10742 matching articles (0.19 per 1,000) in 2006 vs. 2/9840 (0.20 per 1,000) in 2011 [sections: U.S., Politics]
2. "More New York Times opinion pieces or editorials advocating for gender equality in political leadership roles." (source: nyt, predicted: increase)
   Measured: 6/27234 matching articles (0.22 per 1,000) in 2006 vs. 2/19036 (0.11 per 1,000) in 2011 [sections: Opinion, U.S.]
3. "Increase in GDELT events coded as '01 MAKE PUBLIC STATEMENT' or '02 APPEAL' by government or civil society actors promoting women's political participation or leadership." (source: gdelt, predicted: increase)
   Measured: 56377/969651 matching events (58.14 per 1,000) in 2006 vs. 544273/10259907 (53.05 per 1,000) in 2011 [actors: GOV, CVL; location contains: U, n, i, t, e, d,  , S, t, a, t, e, s]
4. "Increase in GDELT events involving protests or demonstrations supporting women's rights or opposing gender discrimination in politics." (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL; location contains: U, n, i, t, e, d,  , S, t, a, t, e, s]
5. "More New York Times coverage of high-profile female political candidates or leaders receiving positive media tone." (source: nyt, predicted: increase)
   Measured: 1/10742 matching articles (0.09 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: Politics, U.S.]

=== Direction B ===
1. "Increase in New York Times headlines or abstracts emphasizing male political leaders' successes or dominance in politics." (source: nyt, predicted: increase)
   Measured: 3/10742 matching articles (0.28 per 1,000) in 2006 vs. 2/9840 (0.20 per 1,000) in 2011 [sections: U.S., Politics]
2. "More New York Times opinion pieces or editorials expressing skepticism about women's effectiveness in political leadership or advocating traditional gender roles." (source: nyt, predicted: increase)
   Measured: 0/27234 matching articles (0.00 per 1,000) in 2006 vs. 0/19036 (0.00 per 1,000) in 2011 [sections: Opinion, U.S.]
3. "Increase in GDELT events coded as '11 DISAPPROVE' or '12 REJECT' involving actors opposing women's political participation or leadership." (source: gdelt, predicted: increase)
   Measured: 19791/969651 matching events (20.41 per 1,000) in 2006 vs. 149969/10259907 (14.62 per 1,000) in 2011 [actors: GOV, REL; location contains: U, n, i, t, e, d,  , S, t, a, t, e, s]
4. "Increase in GDELT events involving protests or demonstrations opposing women's political rights or leadership roles." (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL; location contains: U, n, i, t, e, d,  , S, t, a, t, e, s]
5. "Decrease in New York Times coverage of female political leaders or candidates, or less positive media tone about women in politics." (source: nyt, predicted: decrease)
   Measured: 1/10742 matching articles (0.09 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: Politics, U.S.]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "A", "reasoning": "The evidence shows no increase in media emphasis on male political leadership or skepticism about women leaders, and opposition events toward women's political participation actually decreased. Although some indicators for Direction A show slight decreases or minimal changes, the overall trend in opposition events and opinion pieces does not support Direction B. Therefore, the evidence favors Direction A, that people became less likely to think men make better political leaders than women.", "requested_evidence": []}
```

## 41. G021 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people see themselves as a citizen of their nation, coded 1 (strongly agree) to 4 (strongly disagree). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "strongly disagree" (people became less likely to identify as a citizen of their nation).
Direction B: average answer moved toward "strongly agree" (people became more likely to identify as a citizen of their nation).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines expressing distrust or alienation from the United States or questioning national identity.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["alienation", "national identity crisis", "disillusionment with America", "loss of patriotism", "distrust in government"],
      "nyt_sections": ["U.S.", "Opinion"]
    },
    {
      "indicator": "More GDELT events coded as protests (14) involving civilians expressing dissatisfaction with the government or national institutions.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_location_contains": ["United States"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in New York Times headlines about social fragmentation, regionalism, or movements advocating separation or reduced national unity.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["secession", "regional divide", "fractured nation", "states' rights movement", "national unity challenged"],
      "nyt_sections": ["U.S."]
    },
    {
      "indicator": "More GDELT events involving government officials rejecting or disapproving national policies or symbols (codes 12 REJECT, 11 DISAPPROVE) indicating weakening national consensus.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["11", "12"],
      "gdelt_actor_types": ["GOV"],
      "gdelt_location_contains": ["United States"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in New York Times opinion pieces or editorials discussing decline in patriotism or national pride.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["decline in patriotism", "national pride fading", "identity crisis America", "citizenship questioned"],
      "nyt_sections": ["Opinion"]
    }
  ],
  "B": [
    {
      "indicator": "Increase in New York Times headlines celebrating American patriotism, national unity, or pride in citizenship.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["patriotism", "national pride", "united states of America", "celebrating citizenship", "American unity"],
      "nyt_sections": ["U.S.", "Opinion"]
    },
    {
      "indicator": "More GDELT events involving government or civic leaders making public statements (01 MAKE PUBLIC STATEMENT) promoting national unity or encouraging citizenship pride.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01"],
      "gdelt_actor_types": ["GOV", "EDU", "REL"],
      "gdelt_location_contains": ["United States"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in New York Times coverage of national holidays or events emphasizing American identity and citizenship, such as Independence Day or citizenship ceremonies.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["Independence Day", "citizenship ceremony", "Fourth of July", "national celebration", "American flag"],
      "nyt_sections": ["U.S."]
    },
    {
      "indicator": "More GDELT events coded as appeals (02 APPEAL) or expressions of intent to cooperate (03 EXPRESS INTENT TO COOPERATE) among civic groups or government promoting national solidarity.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["02", "03"],
      "gdelt_actor_types": ["GOV", "CVL", "EDU"],
      "gdelt_location_contains": ["United States"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in New York Times opinion articles or editorials encouraging civic engagement and pride in American citizenship.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["civic pride", "engaged citizenship", "American values", "national identity strengthened"],
      "nyt_sections": ["Opinion"]
    }
  ]
}
```

## 42. G021 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: whether people see themselves as a citizen of their nation, coded 1 (strongly agree) to 4 (strongly disagree). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "strongly disagree" (people became less likely to identify as a citizen of their nation).
Direction B: average answer moved toward "strongly agree" (people became more likely to identify as a citizen of their nation).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines expressing distrust or alienation from the United States or questioning national identity." (source: nyt, predicted: increase)
   Measured: 1/27234 matching articles (0.04 per 1,000) in 2006 vs. 5/19036 (0.26 per 1,000) in 2011 [sections: U.S., Opinion]
2. "More GDELT events coded as protests (14) involving civilians expressing dissatisfaction with the government or national institutions." (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL; location contains: United States]; avg tone 5.51 vs. 6.09
3. "Increase in New York Times headlines about social fragmentation, regionalism, or movements advocating separation or reduced national unity." (source: nyt, predicted: increase)
   Measured: 4/10742 matching articles (0.37 per 1,000) in 2006 vs. 1/9840 (0.10 per 1,000) in 2011 [sections: U.S.]
4. "More GDELT events involving government officials rejecting or disapproving national policies or symbols (codes 12 REJECT, 11 DISAPPROVE) indicating weakening national consensus." (source: gdelt, predicted: increase)
   Measured: 19791/969651 matching events (20.41 per 1,000) in 2006 vs. 149969/10259907 (14.62 per 1,000) in 2011 [actors: GOV; location contains: United States]; avg tone 5.58 vs. 5.47
5. "Increase in New York Times opinion pieces or editorials discussing decline in patriotism or national pride." (source: nyt, predicted: increase)
   Measured: 0/16492 matching articles (0.00 per 1,000) in 2006 vs. 0/9196 (0.00 per 1,000) in 2011 [sections: Opinion]

=== Direction B ===
1. "Increase in New York Times headlines celebrating American patriotism, national unity, or pride in citizenship." (source: nyt, predicted: increase)
   Measured: 20/27234 matching articles (0.73 per 1,000) in 2006 vs. 6/19036 (0.32 per 1,000) in 2011 [sections: U.S., Opinion]
2. "More GDELT events involving government or civic leaders making public statements (01 MAKE PUBLIC STATEMENT) promoting national unity or encouraging citizenship pride." (source: gdelt, predicted: increase)
   Measured: 36105/969651 matching events (37.24 per 1,000) in 2006 vs. 341620/10259907 (33.30 per 1,000) in 2011 [actors: GOV, EDU, REL; location contains: United States]; avg tone 5.74 vs. 5.78
3. "Increase in New York Times coverage of national holidays or events emphasizing American identity and citizenship, such as Independence Day or citizenship ceremonies." (source: nyt, predicted: increase)
   Measured: 9/10742 matching articles (0.84 per 1,000) in 2006 vs. 11/9840 (1.12 per 1,000) in 2011 [sections: U.S.]
4. "More GDELT events coded as appeals (02 APPEAL) or expressions of intent to cooperate (03 EXPRESS INTENT TO COOPERATE) among civic groups or government promoting national solidarity." (source: gdelt, predicted: increase)
   Measured: 42738/969651 matching events (44.08 per 1,000) in 2006 vs. 532381/10259907 (51.89 per 1,000) in 2011 [actors: GOV, CVL, EDU; location contains: United States]; avg tone 6.15 vs. 6.22
5. "Increase in New York Times opinion articles or editorials encouraging civic engagement and pride in American citizenship." (source: nyt, predicted: increase)
   Measured: 8/16492 matching articles (0.49 per 1,000) in 2006 vs. 1/9196 (0.11 per 1,000) in 2011 [sections: Opinion]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "A", "reasoning": "The evidence for Direction A shows a clear increase in distrust or alienation headlines and protests expressing dissatisfaction, while indicators for Direction B mostly show decreases or minimal increases in patriotism-related coverage and events. The decline in government disapproval events and opinion pieces encouraging pride further supports a shift away from national identification.", "requested_evidence": []}
```

## 43. E069_01 -- generate_indicators (cached=True)

**System:**
```
You are a research assistant testing hypotheses about social attitude change using independent, real-world data (news coverage and global event records). You will be given a description of an attitude that shifted between two time points in a country, with two possible directions of the shift — only one is correct, but you are not told which. For each direction, generate concrete, falsifiable indicators of what we would expect to observe in news coverage or world-event data if that direction had occurred. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how much confidence people have in churches, coded 1 (a great deal) to 4 (none at all). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "none at all" (people had less confidence in churches).
Direction B: average answer moved toward "a great deal" (people had more confidence in churches).

For EACH direction, produce at least 5 indicators: concrete, observable signals we could look for in (a) New York Times article headlines/abstracts, or (b) GDELT news-event records (events located in the United States) between 2005 and 2012 that we would expect to see MORE or LESS of if that direction were the true one.

For an NYT indicator, return: `indicator` (one sentence), `source`: "nyt", `expected_change` ("increase" or "decrease"), `nyt_keywords` (phrases to search headlines/abstracts/subject tags for), and optionally `nyt_sections` (restrict to these NYT section or desk names, e.g. "U.S.", "Opinion").

For a GDELT indicator, return: `indicator`, `source`: "gdelt", `expected_change`, and:
- `gdelt_event_codes`: one or more CAMEO codes — a 2-digit root from this list (01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE) or a more specific code within one (e.g. "1411")
- optionally `gdelt_actor_types`: actor categories to require (from: GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public)
- optionally `gdelt_actor_name_contains`: substrings to match in the actors' names (crude — actor-name extraction is inconsistent)
- optionally `gdelt_location_contains`: substrings to match in the event location (e.g. a US state name)
- optionally `gdelt_tone` ("more_negative"/"more_positive"/"no_prediction") and/or `gdelt_goldstein` ("more_negative"/"more_positive"/"no_prediction") — tone is sentiment of the covering articles, Goldstein is the event's own cooperation/conflict intensity
- optionally `gdelt_baseline_codes`: if comparing this indicator's matching events against ALL GDELT events that year risks being misleading because GDELT's total indexed volume changed a lot between 2006 and 2011, give a narrower set of CAMEO root codes to use as the comparison baseline instead (e.g. other political/civic-discourse categories), so the share is of a more comparable denominator

Return JSON: {"A": [...at least 5...], "B": [...at least 5...]}
```

**Response:**
```
{
  "A": [
    {
      "indicator": "Increase in New York Times headlines or abstracts discussing scandals or controversies involving churches or religious leaders.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["church scandal", "religious leader controversy", "church abuse", "clergy misconduct"],
      "nyt_sections": ["U.S.", "National", "Religion"]
    },
    {
      "indicator": "More GDELT events coded as DISAPPROVE (11) or REJECT (12) involving religious actors in the United States.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["11", "12"],
      "gdelt_actor_types": ["REL"],
      "gdelt_location_contains": ["United States"],
      "gdelt_tone": "more_negative",
      "gdelt_goldstein": "more_negative"
    },
    {
      "indicator": "Increase in New York Times coverage highlighting declining church attendance or membership in the U.S.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["declining church attendance", "church membership decline", "religious participation down"],
      "nyt_sections": ["U.S.", "Religion"]
    },
    {
      "indicator": "More GDELT events involving protests (14) or public demonstrations against churches or religious institutions in the U.S.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["14"],
      "gdelt_actor_types": ["CVL"],
      "gdelt_location_contains": ["United States"],
      "gdelt_tone": "more_negative"
    },
    {
      "indicator": "Increase in New York Times opinion pieces or editorials critical of churches or religious institutions.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["church criticism", "religious institution critique", "church accountability"],
      "nyt_sections": ["Opinion", "Religion"]
    }
  ],
  "B": [
    {
      "indicator": "Increase in New York Times headlines or abstracts reporting on positive church community outreach or charitable activities.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["church charity", "religious outreach", "church community service", "faith-based aid"],
      "nyt_sections": ["U.S.", "Religion"]
    },
    {
      "indicator": "More GDELT events coded as PROVIDE AID (07) or ENGAGE IN MATERIAL COOPERATION (06) involving religious actors in the U.S.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["06", "07"],
      "gdelt_actor_types": ["REL"],
      "gdelt_location_contains": ["United States"],
      "gdelt_tone": "more_positive",
      "gdelt_goldstein": "more_positive"
    },
    {
      "indicator": "Increase in New York Times coverage highlighting growth in church attendance or religious affiliation in the U.S.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["increasing church attendance", "religious affiliation growth", "faith participation rise"],
      "nyt_sections": ["U.S.", "Religion"]
    },
    {
      "indicator": "More GDELT events involving public statements (01) or appeals (02) by religious leaders promoting unity or social support.",
      "source": "gdelt",
      "expected_change": "increase",
      "gdelt_event_codes": ["01", "02"],
      "gdelt_actor_types": ["REL"],
      "gdelt_location_contains": ["United States"],
      "gdelt_tone": "more_positive"
    },
    {
      "indicator": "Increase in New York Times opinion pieces or editorials supportive of churches or emphasizing their positive social role.",
      "source": "nyt",
      "expected_change": "increase",
      "nyt_keywords": ["church support", "religious institution praise", "faith community benefits"],
      "nyt_sections": ["Opinion", "Religion"]
    }
  ]
}
```

## 44. E069_01 -- aggregate_judge round 1 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how much confidence people have in churches, coded 1 (a great deal) to 4 (none at all). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "none at all" (people had less confidence in churches).
Direction B: average answer moved toward "a great deal" (people had more confidence in churches).

This is round 1 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines or abstracts discussing scandals or controversies involving churches or religious leaders." (source: nyt, predicted: increase)
   Measured: 1/12293 matching articles (0.08 per 1,000) in 2006 vs. 1/10627 (0.09 per 1,000) in 2011 [sections: U.S., National, Religion]
2. "More GDELT events coded as DISAPPROVE (11) or REJECT (12) involving religious actors in the United States." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL; location contains: United States]
3. "Increase in New York Times coverage highlighting declining church attendance or membership in the U.S." (source: nyt, predicted: increase)
   Measured: 0/10742 matching articles (0.00 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Religion]
4. "More GDELT events involving protests (14) or public demonstrations against churches or religious institutions in the U.S." (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL; location contains: United States]; avg tone 5.51 vs. 6.09
5. "Increase in New York Times opinion pieces or editorials critical of churches or religious institutions." (source: nyt, predicted: increase)
   Measured: 0/16492 matching articles (0.00 per 1,000) in 2006 vs. 0/9196 (0.00 per 1,000) in 2011 [sections: Opinion, Religion]

=== Direction B ===
1. "Increase in New York Times headlines or abstracts reporting on positive church community outreach or charitable activities." (source: nyt, predicted: increase)
   Measured: 0/10742 matching articles (0.00 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Religion]
2. "More GDELT events coded as PROVIDE AID (07) or ENGAGE IN MATERIAL COOPERATION (06) involving religious actors in the U.S." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL; location contains: United States]
3. "Increase in New York Times coverage highlighting growth in church attendance or religious affiliation in the U.S." (source: nyt, predicted: increase)
   Measured: 0/10742 matching articles (0.00 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Religion]
4. "More GDELT events involving public statements (01) or appeals (02) by religious leaders promoting unity or social support." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL; location contains: United States]
5. "Increase in New York Times opinion pieces or editorials supportive of churches or emphasizing their positive social role." (source: nyt, predicted: increase)
   Measured: 0/16492 matching articles (0.00 per 1,000) in 2006 vs. 0/9196 (0.00 per 1,000) in 2011 [sections: Opinion, Religion]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "need_more_evidence", "reasoning": "The indicators for both directions show almost no change or very minimal activity in the predicted directions, with most measures at zero or near zero for both 2006 and 2011. The only notable increase is in protests against churches (Direction A), but the tone is slightly more positive, which complicates interpretation. Overall, the evidence is insufficient to decisively favor either direction.", "requested_evidence": [{"direction": "A", "indicator": "Increase in nationally representative survey data on reported church attendance or membership", "source": "survey", "expected_change": "decrease"}, {"direction": "A", "indicator": "Increase in negative social media sentiment or volume of posts about churches or religious leaders", "source": "social_media", "expected_change": "increase"}, {"direction": "B", "indicator": "Increase in nationally representative survey data on perceived positive impact of churches in communities", "source": "survey", "expected_change": "increase"}, {"direction": "B", "indicator": "Increase in charitable donations to churches or religious organizations", "source": "financial_records", "expected_change": "increase"}, {"direction": "A", "indicator": "Increase in reported scandals or legal cases involving churches or religious leaders in major news outlets", "source": "nyt", "expected_change": "increase", "nyt_keywords": ["scandal", "abuse", "investigation", "church", "religious leader"], "nyt_sections": ["U.S.", "National", "Religion"]}]}
```

## 45. E069_01 -- aggregate_judge round 2 (cached=True)

**System:**
```
You are evaluating competing hypotheses about a social attitude shift, using indicators that have already been checked against real-world evidence (news coverage and event records). You will be given both possible directions of the shift — only one is correct, but you are not told which — plus, for each direction, a set of indicators and what was actually measured for each. Weigh the indicators by how directly and strongly each result bears on the question (a large, clean move in the predicted direction counts more than a tiny or ambiguous one). Decide which direction the evidence favors overall, or, if you judge the evidence insufficient, request specific additional indicators to check. Respond only in the requested JSON format.
```

**User:**
```
Country: United States
Time points: 2006 and 2011
Attitude measured: how much confidence people have in churches, coded 1 (a great deal) to 4 (none at all). A nationally representative survey asked this question at both time points.

Between 2006 and 2011, the average answer moved in one of these two directions (you are not told which actually happened):
Direction A: average answer moved toward "none at all" (people had less confidence in churches).
Direction B: average answer moved toward "a great deal" (people had more confidence in churches).

This is round 2 of at most 3. If the evidence below is not yet sufficient to decide, you may request up to 5 additional specific indicators to check instead of deciding now.

Indicators checked so far, with what was actually measured:

=== Direction A ===
1. "Increase in New York Times headlines or abstracts discussing scandals or controversies involving churches or religious leaders." (source: nyt, predicted: increase)
   Measured: 1/12293 matching articles (0.08 per 1,000) in 2006 vs. 1/10627 (0.09 per 1,000) in 2011 [sections: U.S., National, Religion]
2. "More GDELT events coded as DISAPPROVE (11) or REJECT (12) involving religious actors in the United States." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL; location contains: United States]
3. "Increase in New York Times coverage highlighting declining church attendance or membership in the U.S." (source: nyt, predicted: increase)
   Measured: 0/10742 matching articles (0.00 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Religion]
4. "More GDELT events involving protests (14) or public demonstrations against churches or religious institutions in the U.S." (source: gdelt, predicted: increase)
   Measured: 274/969651 matching events (0.28 per 1,000) in 2006 vs. 3731/10259907 (0.36 per 1,000) in 2011 [actors: CVL; location contains: United States]; avg tone 5.51 vs. 6.09
5. "Increase in New York Times opinion pieces or editorials critical of churches or religious institutions." (source: nyt, predicted: increase)
   Measured: 0/16492 matching articles (0.00 per 1,000) in 2006 vs. 0/9196 (0.00 per 1,000) in 2011 [sections: Opinion, Religion]
6. "Increase in reported scandals or legal cases involving churches or religious leaders in major news outlets" (source: nyt, predicted: increase)
   Measured: 854/12293 matching articles (69.47 per 1,000) in 2006 vs. 217/10627 (20.42 per 1,000) in 2011 [sections: U.S., National, Religion]

=== Direction B ===
1. "Increase in New York Times headlines or abstracts reporting on positive church community outreach or charitable activities." (source: nyt, predicted: increase)
   Measured: 0/10742 matching articles (0.00 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Religion]
2. "More GDELT events coded as PROVIDE AID (07) or ENGAGE IN MATERIAL COOPERATION (06) involving religious actors in the U.S." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL; location contains: United States]
3. "Increase in New York Times coverage highlighting growth in church attendance or religious affiliation in the U.S." (source: nyt, predicted: increase)
   Measured: 0/10742 matching articles (0.00 per 1,000) in 2006 vs. 0/9840 (0.00 per 1,000) in 2011 [sections: U.S., Religion]
4. "More GDELT events involving public statements (01) or appeals (02) by religious leaders promoting unity or social support." (source: gdelt, predicted: increase)
   Measured: 0/969651 matching events (0.00 per 1,000) in 2006 vs. 0/10259907 (0.00 per 1,000) in 2011 [actors: REL; location contains: United States]
5. "Increase in New York Times opinion pieces or editorials supportive of churches or emphasizing their positive social role." (source: nyt, predicted: increase)
   Measured: 0/16492 matching articles (0.00 per 1,000) in 2006 vs. 0/9196 (0.00 per 1,000) in 2011 [sections: Opinion, Religion]

Decide which direction the evidence favors overall, or request specific additional indicators to check.
Respond only in this JSON format: {"decision": "A"|"B"|"need_more_evidence", "reasoning": "<2-3 sentences weighing the evidence>", "requested_evidence": [{"direction": "A"|"B", "indicator": "...", "source": "nyt", "expected_change": "increase"|"decrease", "nyt_keywords": [...], "nyt_sections": [...] (optional)} or {"direction": "A"|"B", "indicator": "...", "source": "gdelt", "expected_change": "increase"|"decrease", "gdelt_event_codes": [...] (2-digit root from 01 MAKE PUBLIC STATEMENT, 02 APPEAL, 03 EXPRESS INTENT TO COOPERATE, 04 CONSULT, 05 ENGAGE IN DIPLOMATIC COOPERATION, 06 ENGAGE IN MATERIAL COOPERATION, 07 PROVIDE AID, 08 YIELD, 09 INVESTIGATE, 10 DEMAND, 11 DISAPPROVE, 12 REJECT, 13 THREATEN, 14 PROTEST, 15 EXHIBIT FORCE POSTURE, 16 REDUCE RELATIONS, 17 COERCE, 18 ASSAULT, 19 FIGHT, 20 USE UNCONVENTIONAL MASS VIOLENCE, or a more specific code), "gdelt_actor_types": [...] (optional, from GOV government, LEG legislature, JUD judiciary, COP police, MED media, BUS business, EDU education, REL religious, CVL civilians/general public), "gdelt_actor_name_contains": [...] (optional), "gdelt_location_contains": [...] (optional), "gdelt_tone": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_goldstein": "more_negative"|"more_positive"|"no_prediction" (optional), "gdelt_baseline_codes": [...] (optional)}]}
If decision is not "need_more_evidence", "requested_evidence" must be [].
```

**Response:**
```
{"decision": "A", "reasoning": "The evidence strongly favors Direction A, indicating decreased confidence in churches. Key indicators such as a large decrease in reported scandals or legal cases involving churches (from 69.47 to 20.42 per 1,000 articles) and an increase in protests against religious institutions support this. Other indicators for Direction A show slight or no increases, but none for Direction B show any increase, all being zero, indicating no evidence for increased confidence.", "requested_evidence": []}
```
