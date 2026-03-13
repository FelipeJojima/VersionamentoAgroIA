
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults



pergunta = "sigatoka"

ferramenta = DuckDuckGoSearchResults(output_format="json", max_results=10)

results = ferramenta.invoke(f"{pergunta} site:https://sistemas.epagri.sc.gov.br/sedimob")

print(results)