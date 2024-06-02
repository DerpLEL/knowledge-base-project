from entity_extractor import RefinedEntityExtractor

extractor = RefinedEntityExtractor()

query = "how campuses does ho chi minh city university of technology have?"
result = extractor(query)

for entity, entity_title in result:
    print("Entity:", entity, " | Entity title:", entity_title)
