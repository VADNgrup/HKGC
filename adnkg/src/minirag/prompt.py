GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PROMPTS["DEFAULT_ENTITY_TYPES"] = ["person", "place", "time", "event", "artifact", "object", "concept"]


PROMPTS["entity_extraction"] = """-Goal-
Given a text document potentially relevant to historical and domain-specific context (specifically focused on history) and a list of entity types, identify all entities of those types from the text, including their alternate names, temporal range and geographical location, and all relationships among them.

-Steps-
1. Identify all entities. For each identified entity, extract the following:
- entity_name: Clean, canonical name in uppercase. Strip all HTML tags, brackets, and outer quotes. PRESERVE VIETNAMESE ACCENTS/DIACRITICS (e.g. 'TRẦN HƯNG ĐẠO', 'ĐẠI VIỆT', 'THÁI BÌNH').
- entity_type: One of the following: [{entity_types}] (Focus on historical individuals (person), historical sites/locations/states (place), specific dates/eras/dynasties (time), historical occurrences/wars/treaties (event), historical relics/books/wills (artifact), historical objects/weapons (object), and historical concepts/systems/policies (concept)).
- alternate_names: Semicolon-separated list of alternate names, aliases, reign names, courtesy names, titles, or birth names (e.g. 'TRẦN QUỐC TUẤN; HƯNG ĐẠO ĐẠI VƯƠNG'). If none exist, output 'none'.
- entity_description: Comprehensive summary of attributes, roles, and activities.
- time_range: Start and end years or period (e.g. '1225-1258', '1370', 'Thế kỷ 13', or 'unknown').
- geo_location: Geographical coordinates, region, or 'unknown'.

Format each entity as: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<alternate_names>{tuple_delimiter}<entity_description>{tuple_delimiter}<time_range>{tuple_delimiter}<geo_location>)

CRITICAL STRICT RULES FOR ENTITY EXTRACTION:
- TRANSLATE CONCEPT/EVENT TYPES TO ENGLISH: Translate general concept words, roles, and event types to standard English (e.g. 'Chiến tranh' -> 'WAR', 'Xây dựng' -> 'CONSTRUCTION', 'Đại tướng quân' -> 'GENERAL').
- PRESERVE PROPER NOUNS & ACCENTS: Keep historical proper nouns (like 'ĐẠI VIỆT', 'THĂNG LONG', 'TRẦN THÁI TÔNG') clean and canonical WITH PRESERVED VIETNAMESE ACCENTS/DIACRITICS.
- NO CSS/WEB ARTIFACTS OR NOISE: Absolutely NEVER extract CSS properties, Wikidata/Wikimedia artifacts, website UI text (e.g. 'Wikidata', 'Wayback Machine', 'Đăng nhập'), or long base64 strings.
- SEPARATE ALTERNATIVE NAMES: Put all alternative names into the <alternate_names> field (e.g. instead of name='HUỆ TÚC VƯƠNG (TRẦN ĐẠI NIÊN)', put entity_name='HUỆ TÚC VƯƠNG' and alternate_names='TRẦN ĐẠI NIÊN').

2. Identify related pairs of entities. For each pair, extract:
- source_entity: Name of the source entity, matching step 1.
- target_entity: Name of the target entity, matching step 1.
- relationship_description: Detailed explanation of their relationship.
- relationship_keywords: One or more comma-separated keywords capturing the nature of the relationship.
- relationship_strength: Numeric score (1-10) indicating strength of connection.
- time_range: Time range when this relationship was active (e.g. '1225-1258', '1370', or 'unknown').

Format each relationship as: ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>{tuple_delimiter}<time_range>)

3. Identify overarching keywords summarizing the main concepts, themes, or topics of the entire text.
Format the content keywords as: ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output as a single list of all entities and relationships, using **{record_delimiter}** as the delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:
Entity_types: [person, place, time, event, artifact, object, concept]
Text:
In 1066, the Norman-French army led by William, Duke of Normandy, invaded England. King Harold Godwinson led the Anglo-Saxon forces against them at the Battle of Hastings near Senlac Hill, resulting in a decisive Norman victory.
###############
Output:
("entity"{tuple_delimiter}"WILLIAM THE CONQUEROR"{tuple_delimiter}"person"{tuple_delimiter}"WILLIAM I; DUKE OF NORMANDY"{tuple_delimiter}"Duke of Normandy who led the Norman conquest of England."{tuple_delimiter}"1035-1087"{tuple_delimiter}"Normandy, England"){record_delimiter}
("entity"{tuple_delimiter}"HAROLD GODWINSON"{tuple_delimiter}"person"{tuple_delimiter}"HAROLD II"{tuple_delimiter}"The last crowned Anglo-Saxon king of England, died at the Battle of Hastings."{tuple_delimiter}"1022-1066"{tuple_delimiter}"England"){record_delimiter}
("entity"{tuple_delimiter}"BATTLE OF HASTINGS"{tuple_delimiter}"event"{tuple_delimiter}"none"{tuple_delimiter}"A decisive battle fought between the Norman-French army and the Anglo-Saxons."{tuple_delimiter}"1066"{tuple_delimiter}"Hastings, England"){record_delimiter}
("entity"{tuple_delimiter}"ENGLAND"{tuple_delimiter}"place"{tuple_delimiter}"none"{tuple_delimiter}"A sovereign country on the island of Great Britain."{tuple_delimiter}"unknown"{tuple_delimiter}"Great Britain"){record_delimiter}
("entity"{tuple_delimiter}"NORMANDY"{tuple_delimiter}"place"{tuple_delimiter}"none"{tuple_delimiter}"A geographical region in northwestern France."{tuple_delimiter}"unknown"{tuple_delimiter}"France"){record_delimiter}
("relationship"{tuple_delimiter}"WILLIAM THE CONQUEROR"{tuple_delimiter}"BATTLE OF HASTINGS"{tuple_delimiter}"William led the Norman forces to victory at Hastings."{tuple_delimiter}"led, military victory"{tuple_delimiter}9{tuple_delimiter}"1066"){record_delimiter}
("relationship"{tuple_delimiter}"HAROLD GODWINSON"{tuple_delimiter}"BATTLE OF HASTINGS"{tuple_delimiter}"Harold led the Anglo-Saxon forces and died in the battle."{tuple_delimiter}"led, military defeat, death"{tuple_delimiter}9{tuple_delimiter}"1066"){record_delimiter}
("relationship"{tuple_delimiter}"WILLIAM THE CONQUEROR"{tuple_delimiter}"ENGLAND"{tuple_delimiter}"William invaded England and became its king."{tuple_delimiter}"invaded, ruled"{tuple_delimiter}8{tuple_delimiter}"1066-1087"){record_delimiter}
("relationship"{tuple_delimiter}"BATTLE OF HASTINGS"{tuple_delimiter}"ENGLAND"{tuple_delimiter}"The battle took place in England near Hastings."{tuple_delimiter}"location"{tuple_delimiter}9{tuple_delimiter}"1066"){record_delimiter}
("content_keywords"{tuple_delimiter}"Norman conquest, Battle of Hastings, William the Conqueror, Anglo-Saxon England"){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""


PROMPTS[
    "entiti_continue_extraction_mini"
] = """MANY entities were missed in the last extraction.
After summarizing with all the information previously extracted, compared to the original text, it was noticed that the following information was mainly omitted:
{omit}

The types of entities that need to be added can be obtained from Entity_types,
or you can add them yourself.

Entity_types: {entity_types}


Add them below using the same format:
"""


PROMPTS["minirag_query2kwd"] = """---Role---

You are a helpful assistant tasked with identifying both answer-type and low-level keywords in the user's query.

---Goal---

Given the query, list both answer-type and low-level keywords.
answer_type_keywords focus on the type of the answer to the certain query, while low-level keywords focus on specific entities, details, or concrete terms.
The answer_type_keywords must be selected from Answer type pool.
This pool is in the form of a dictionary, where the key represents the Type you should choose from and the value represents the example samples.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have three keys:
  - "answer_type_keywords" for the types of the answer. In this list, the types with the highest likelihood should be placed at the forefront. No more than 3.
  - "entities_from_query" for specific entities or details. It must be extracted from the query.
######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
Answer type pool: {{
 'PERSONAL LIFE': ['FAMILY TIME', 'HOME MAINTENANCE'],
 'STRATEGY': ['MARKETING PLAN', 'BUSINESS EXPANSION'],
 'SERVICE FACILITATION': ['ONLINE SUPPORT', 'CUSTOMER SERVICE TRAINING'],
 'PERSON': ['JANE DOE', 'JOHN SMITH'],
 'FOOD': ['PASTA', 'SUSHI'],
 'EMOTION': ['HAPPINESS', 'ANGER'],
 'PERSONAL EXPERIENCE': ['TRAVEL ABROAD', 'STUDYING ABROAD'],
 'INTERACTION': ['TEAM MEETING', 'NETWORKING EVENT'],
 'BEVERAGE': ['COFFEE', 'TEA'],
 'PLAN': ['ANNUAL BUDGET', 'PROJECT TIMELINE'],
 'GEO': ['NEW YORK CITY', 'SOUTH AFRICA'],
 'GEAR': ['CAMPING TENT', 'CYCLING HELMET'],
 'EMOJI': ['ðŸŽ‰', 'ðŸš€'],
 'BEHAVIOR': ['POSITIVE FEEDBACK', 'NEGATIVE CRITICISM'],
 'TONE': ['FORMAL', 'INFORMAL'],
 'LOCATION': ['DOWNTOWN', 'SUBURBS']
}}
################
Output:
{{
  "answer_type_keywords": ["STRATEGY","PERSONAL LIFE"],
  "entities_from_query": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "When was SpaceX's first rocket launch?"
Answer type pool: {{
 'DATE AND TIME': ['2023-10-10 10:00', 'THIS AFTERNOON'],
 'ORGANIZATION': ['GLOBAL INITIATIVES CORPORATION', 'LOCAL COMMUNITY CENTER'],
 'PERSONAL LIFE': ['DAILY EXERCISE ROUTINE', 'FAMILY VACATION PLANNING'],
 'STRATEGY': ['NEW PRODUCT LAUNCH', 'YEAR-END SALES BOOST'],
 'SERVICE FACILITATION': ['REMOTE IT SUPPORT', 'ON-SITE TRAINING SESSIONS'],
 'PERSON': ['ALEXANDER HAMILTON', 'MARIA CURIE'],
 'FOOD': ['GRILLED SALMON', 'VEGETARIAN BURRITO'],
 'EMOTION': ['EXCITEMENT', 'DISAPPOINTMENT'],
 'PERSONAL EXPERIENCE': ['BIRTHDAY CELEBRATION', 'FIRST MARATHON'],
 'INTERACTION': ['OFFICE WATER COOLER CHAT', 'ONLINE FORUM DEBATE'],
 'BEVERAGE': ['ICED COFFEE', 'GREEN SMOOTHIE'],
 'PLAN': ['WEEKLY MEETING SCHEDULE', 'MONTHLY BUDGET OVERVIEW'],
 'GEO': ['MOUNT EVEREST BASE CAMP', 'THE GREAT BARRIER REEF'],
 'GEAR': ['PROFESSIONAL CAMERA EQUIPMENT', 'OUTDOOR HIKING GEAR'],
 'EMOJI': ['ðŸ“…', 'â�°'],
 'BEHAVIOR': ['PUNCTUALITY', 'HONESTY'],
 'TONE': ['CONFIDENTIAL', 'SATIRICAL'],
 'LOCATION': ['CENTRAL PARK', 'DOWNTOWN LIBRARY']
}}

################
Output:
{{
  "answer_type_keywords": ["DATE AND TIME", "ORGANIZATION", "PLAN"],
  "entities_from_query": ["SpaceX", "Rocket launch", "Aerospace", "Power Recovery"]

}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
Answer type pool: {{
 'PERSONAL LIFE': ['MANAGING WORK-LIFE BALANCE', 'HOME IMPROVEMENT PROJECTS'],
 'STRATEGY': ['MARKETING STRATEGIES FOR Q4', 'EXPANDING INTO NEW MARKETS'],
 'SERVICE FACILITATION': ['CUSTOMER SATISFACTION SURVEYS', 'STAFF RETENTION PROGRAMS'],
 'PERSON': ['ALBERT EINSTEIN', 'MARIA CALLAS'],
 'FOOD': ['PAN-FRIED STEAK', 'POACHED EGGS'],
 'EMOTION': ['OVERWHELM', 'CONTENTMENT'],
 'PERSONAL EXPERIENCE': ['LIVING ABROAD', 'STARTING A NEW JOB'],
 'INTERACTION': ['SOCIAL MEDIA ENGAGEMENT', 'PUBLIC SPEAKING'],
 'BEVERAGE': ['CAPPUCCINO', 'MATCHA LATTE'],
 'PLAN': ['ANNUAL FITNESS GOALS', 'QUARTERLY BUSINESS REVIEW'],
 'GEO': ['THE AMAZON RAINFOREST', 'THE GRAND CANYON'],
 'GEAR': ['SURFING ESSENTIALS', 'CYCLING ACCESSORIES'],
 'EMOJI': ['ðŸ’»', 'ðŸ“±'],
 'BEHAVIOR': ['TEAMWORK', 'LEADERSHIP'],
 'TONE': ['FORMAL MEETING', 'CASUAL CONVERSATION'],
 'LOCATION': ['URBAN CITY CENTER', 'RURAL COUNTRYSIDE']
}}

################
Output:
{{
  "answer_type_keywords": ["STRATEGY", "PERSON"],
  "entities_from_query": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
Example 4:

Query: "Where is the capital of the United States?"
Answer type pool: {{
 'ORGANIZATION': ['GREENPEACE', 'RED CROSS'],
 'PERSONAL LIFE': ['DAILY WORKOUT', 'HOME COOKING'],
 'STRATEGY': ['FINANCIAL INVESTMENT', 'BUSINESS EXPANSION'],
 'SERVICE FACILITATION': ['ONLINE SUPPORT', 'CUSTOMER SERVICE TRAINING'],
 'PERSON': ['ALBERTA SMITH', 'BENJAMIN JONES'],
 'FOOD': ['PASTA CARBONARA', 'SUSHI PLATTER'],
 'EMOTION': ['HAPPINESS', 'SADNESS'],
 'PERSONAL EXPERIENCE': ['TRAVEL ADVENTURE', 'BOOK CLUB'],
 'INTERACTION': ['TEAM BUILDING', 'NETWORKING MEETUP'],
 'BEVERAGE': ['LATTE', 'GREEN TEA'],
 'PLAN': ['WEIGHT LOSS', 'CAREER DEVELOPMENT'],
 'GEO': ['PARIS', 'NEW YORK'],
 'GEAR': ['CAMERA', 'HEADPHONES'],
 'EMOJI': ['ðŸ�¢', 'ðŸŒ�'],
 'BEHAVIOR': ['POSITIVE THINKING', 'STRESS MANAGEMENT'],
 'TONE': ['FRIENDLY', 'PROFESSIONAL'],
 'LOCATION': ['DOWNTOWN', 'SUBURBS']
}}
################
Output:
{{
  "answer_type_keywords": ["LOCATION"],
  "entities_from_query": ["capital of the United States", "Washington", "New York"]
}}
#############################

-Real Data-
######################
Query: {query}
Answer type pool:{TYPE_POOL}
######################
Output:

"""


PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""
