# Embedding clusters of model reasoning (text-embedding-3-small, k-means, k=6)

Shares are the percent of each model's replies in the cluster. Examples are the replies closest to the cluster centre.

## Cluster 1: upheaval, revolution, political upheaval, sisi, subsequent, state, economic, security, instability, political

736 replies, accuracy 57.5%

| model | share % |
|---|---|
| gpt-4.1-mini | 21.1 |
| gpt-5 (reasoning off) | 21.1 |
| gpt-5 (reasoning on) | 21.2 |
| gpt-5-mini (reasoning off) | 21.1 |
| gpt-5-mini (reasoning on) | 21.1 |

- *gpt-5-mini (reasoning off), EGY 6->7 E069_08:* Between 2013 and 2018 Egypt experienced political upheaval (2013 coup, then consolidation under Sisi) with strong state emphasis on stability and public sector roles; public confidence in civil services likely declined somewhat due to perceptions of inefficiency, corruption, and crackdown on dissent. Overall the typical response probably shifted toward less confidence.
- *gpt-5 (reasoning off), EGY 6->7 G007_18_B:* After 2013, Egypt experienced prolonged political instability, economic strain (currency devaluation, inflation), and security concerns, which generally erode social trust beyond immediate circles. By 2018, these pressures likely nudged attitudes toward greater caution even within neighborhoods.
- *gpt-4.1-mini, EGY 5->6 E037:* Between 2008 and 2013, Egypt experienced significant political upheaval, including the 2011 revolution that led to the ousting of Hosni Mubarak and a period of instability. During this time, many Egyptians likely became more skeptical of government effectiveness and may have shifted toward believing that individuals and communities should take more responsibility rather than relying on the government.

## Cluster 4: latin, america, latin america, progressive, strong, pandemic, countries, relatively, stable, policies

649 replies, accuracy 59.6%

| model | share % |
|---|---|
| gpt-4.1-mini | 18.8 |
| gpt-5 (reasoning off) | 18.7 |
| gpt-5 (reasoning on) | 18.5 |
| gpt-5-mini (reasoning off) | 18.8 |
| gpt-5-mini (reasoning on) | 18.4 |

- *gpt-5 (reasoning off), URY 6->7 A173:* Uruguay experienced steady economic growth, expanded social rights (e.g., marriage equality, cannabis regulation), and relatively strong institutions through the 2010s, which likely bolstered perceptions of personal agency. Despite pandemic shocks by 2022, Uruguay managed COVID comparatively well in the region, supporting a shift toward greater perceived control.
- *gpt-4.1-mini, URY 6->7 E037:* Between 2011 and 2022, Uruguay experienced continued social and economic development, with a generally stable political environment and progressive policies that often emphasized social welfare and government involvement in public services. Given this context, it is likely that the population's views shifted toward expecting more government responsibility in addressing social and economic issues.
- *gpt-5 (reasoning off), URY 6->7 E023:* Uruguay experienced significant political engagement in the 2010s, including the legalization reforms under Mujica and subsequent economic and social debates; however, by the early 2020s there was rising disillusionment and lower participation trends across Latin America, with Uruguay seeing some cooling of enthusiasm after reform peaks. Overall, typical responses likely shifted slightly away from strong interest.

## Cluster 3: umbrella, umbrella movement, movement, tensions, mainland, political tensions, rising, beijing, increasing, housing

583 replies, accuracy 58.0%

| model | share % |
|---|---|
| gpt-4.1-mini | 16.4 |
| gpt-5 (reasoning off) | 16.4 |
| gpt-5 (reasoning on) | 16.8 |
| gpt-5-mini (reasoning off) | 16.4 |
| gpt-5-mini (reasoning on) | 17.8 |

- *gpt-4.1-mini, HKG 5->6 E124:* Between 2005 and 2014, Hong Kong experienced increasing political tensions and concerns over Beijing's influence on its autonomy and civil liberties, especially after events like the 2014 Umbrella Movement protests. These developments likely led to a growing perception among the population that respect for individual human rights was declining.
- *gpt-5-mini (reasoning off), HKG 6->7 E233A:* Between 2014 and 2018 Hong Kong experienced major political events (the 2014 Umbrella Movement and growing polarization afterward) that increased emphasis on political rights and reduced support for state-led redistribution among many pro-business and pro-establishment segments; overall, the typical respondent likely became less inclined to view enforced income equality as an essential democratic characteristic.
- *gpt-5 (reasoning off), HKG 6->7 E018:* Between 2014 and 2018, Hong Kong experienced growing political tensions, including the aftermath of the 2014 Umbrella Movement and increasing concerns over Beijing’s influence. This likely made “greater respect for authority” less appealing to many, shifting sentiment toward viewing it as a bad thing.

## Cluster 5: military, coup, military coup, political turmoil, protests, turmoil, political, rule, junta, turmoil including

548 replies, accuracy 44.7%

| model | share % |
|---|---|
| gpt-4.1-mini | 15.1 |
| gpt-5 (reasoning off) | 14.6 |
| gpt-5 (reasoning on) | 16.6 |
| gpt-5-mini (reasoning off) | 15.8 |
| gpt-5-mini (reasoning on) | 16.5 |

- *gpt-4.1-mini, THA 5->6 E069_11:* Between 2007 and 2013, Thailand experienced significant political turmoil, including a military coup in 2006, ongoing protests, and changes in government leadership. These events likely eroded public trust in the government, especially in the capital, Bangkok, where much of the political conflict was centered. Therefore, the typical answer likely moved toward less confidence in the government.
- *gpt-5-mini (reasoning off), THA 5->6 E235:* Between 2007 and 2013 Thailand experienced major political turmoil, including the 2006 coup's aftermath, the 2008-2010 protests, and the 2013-2014 protests that increased polarization; overall public emphasis on democratic governance remained high or rose as competing factions mobilized around legitimacy and representation. Thus the typical response likely shifted toward stronger support for living in a democratically governed country.
- *gpt-4.1-mini, THA 5->6 E236:* Between 2007 and 2013, Thailand experienced significant political turmoil, including a military coup in 2006 and ongoing conflicts between pro- and anti-government groups. By 2013, political instability and military influence likely led to a decline in public perception of democratic governance, moving attitudes away from viewing the country as completely democratic.

## Cluster 0: conflict, euromaidan, orange, orange revolution, russia, war, corruption, european, crisis, revolution

539 replies, accuracy 54.2%

| model | share % |
|---|---|
| gpt-4.1-mini | 15.5 |
| gpt-5 (reasoning off) | 15.5 |
| gpt-5 (reasoning on) | 15.4 |
| gpt-5-mini (reasoning off) | 15.5 |
| gpt-5-mini (reasoning on) | 15.5 |

- *gpt-5-mini (reasoning off), UKR 6->7 E069_04:* Between 2011 and 2020 Ukraine experienced major political upheavals (Euromaidan 2013–2014), conflict with Russia, and intense media polarization; trust in institutions including the press generally declined as media became seen as partisan or influenced by oligarchs. Therefore the typical response moved toward less confidence.
- *gpt-4.1-mini, UKR 6->7 A165:* Between 2011 and 2020, Ukraine experienced significant political and social upheaval, including the Euromaidan protests, the annexation of Crimea by Russia, and ongoing conflict in the eastern regions. These events likely increased societal distrust and caution among the population, leading more people to feel that they need to be very careful in dealing with others.
- *gpt-4.1-mini, UKR 6->7 E018:* Between 2011 and 2020, Ukraine experienced significant political upheaval, including the Euromaidan protests, the 2014 revolution, and ongoing conflict with Russia. These events generally fostered skepticism toward authority figures, especially those perceived as corrupt or oppressive. Therefore, the typical Ukrainian attitude likely shifted toward viewing greater respect for authority less favorably.

## Cluster 2: urbanization, buddhist, exposure, predominantly buddhist, rapid, predominantly, neighbors, younger, urban, education

430 replies, accuracy 54.2%

| model | share % |
|---|---|
| gpt-4.1-mini | 13.2 |
| gpt-5 (reasoning off) | 13.8 |
| gpt-5 (reasoning on) | 11.5 |
| gpt-5-mini (reasoning off) | 12.5 |
| gpt-5-mini (reasoning on) | 10.8 |

- *gpt-4.1-mini, THA 5->6 G023:* Between 2007 and 2013, Thailand experienced significant political and social changes, including increased globalization and exposure to individualistic values through media and education. These factors likely encouraged a stronger sense of individual autonomy among the population over time.
- *gpt-4.1-mini, THA 6->7 A106C:* Between 2013 and 2018, Thailand experienced ongoing social and political changes, including increased community awareness and grassroots movements, which often encourage participation in local organizations like self-help and mutual aid groups. However, urbanization and modernization trends might also reduce traditional group memberships. Overall, the balance likely shifted slightly toward more active engagement as people sought community support amid social changes.
- *gpt-5 (reasoning on), THA 6->7 F135A:* Thailand has seen gradual liberalization of social attitudes, especially among younger, urban, and more internet-connected cohorts between 2013 and 2018. These shifts typically increase tolerance of premarital sex, nudging average responses toward greater justifiability.
