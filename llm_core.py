import openai
import json
from pydantic import BaseModel, Field
from typing import List
from openai import AuthenticationError, LengthFinishReasonError


# ── Schemas do LLM 1 (Extração de Schema) ──────────────────────────────────

class UnitSchema(BaseModel):
    field: str = Field(..., description="Campo/Chave da informação no payload final.")
    is_fixed: bool = Field(..., description="True se o valor for uma constante estática (ex: resource_type='pix_payment'). False se for variável/dinâmico dependendo do input.")
    fixed_value: str | int | None = Field(..., description="Se is_fixed for True, a LLM 1 extrai e preenche o valor exato aqui. Se False, retorna null.")
    type_hint: str = Field(..., description="Se is_fixed for False, instrução de formato para a LLM 2 gerar o dado (ex: 'float', 'nome', 'string').")

class Payload(BaseModel):
    name: str = Field(..., description="Nome do agent/tool associado a esse payload.")
    nodes: List[UnitSchema] = Field(..., description="Lista de chave-valor que compõem o payload.")

class Schema(BaseModel):
    payloads: List[Payload] = Field(..., description="Lista de payloads associados a estrutura/schema identificado.")

class SchemaResponse(BaseModel):
    structure: Schema = Field(..., description="Lista de payloads associados a estrutura/schema identificado.")


# ── Schemas do LLM 2 (Geração de Testes) ───────────────────────────────────

class UnitPayload(BaseModel):
    field: str = Field(..., description="Campo/Chave da informação no payload final.")
    value: str | int = Field(..., description="Valor associado àquele campo.")

class ResponsePayload(BaseModel):
    name: str = Field(..., description="Nome do agent/tool associado a esse payload.")
    nodes: List[UnitPayload] = Field(..., description="Lista de chave-valor que compõem o payload.")

class ResponseUnit(BaseModel):
    user_input: str = Field(..., description="O texto/input que o usuário enviou")
    expected_payload: ResponsePayload = Field(..., description="Payload esperado associado ao input do usuário")

class ResponseFormat(BaseModel):
    response: List[ResponseUnit] = Field(..., description="Dados para teste estruturados no schema")


# ── Schema do Validador ─────────────────────────────────────────────────────

class LogSchema(BaseModel):
    user_input: str = Field(..., description="Nome do agent/tool associado a esse payload.")
    expected_payload: ResponsePayload = Field(..., description="Payload ESPERADO associado ao input do usuário")
    actual_payload: ResponsePayload = Field(..., description="Payload RECEBIDO associado ao input do usuário")

class LogResponse(BaseModel):
    logs: List[LogSchema] = Field(..., description="Dados dos logs análisados e estruturados")

class ValidationResult(BaseModel):
    is_test_flawed: bool = Field(..., description="True se o dado de teste gerado for injusto, impossível ou contraditório. False se o teste for válido.")
    is_llm_flawed: bool = Field(..., description="True se o teste era válido, mas o LLM do sistema principal errou a execução. False se a culpa for do teste.")
    diagnosis: str = Field(..., description="Explicação técnica curta do motivo da falha e quem foi o culpado.")
    suggested_action: str = Field(..., description="Se a culpa for do teste, sugira como corrigir o teste. Se for do LLM, indique qual regra faltou no prompt principal.")


# ── Schema do Refiner─────────────────────────────────────────────────────

class PromptRef(BaseModel):
    prompt: str = Field(..., description="Novo prompt refinado e melhorado")
    changes: str = Field(..., description="Mudanças do antigo para o novo prompt")

# ── Prompts ─────────────────────────────────────────────────────────────────

PROMPT_TESTER_1 = """
## Missão e Identificação
Você é o Max Tester, um Arquiteto de Dados especialista em engenharia reversa de APIs e análise de logs de LLMs.
Seu objetivo é analisar exemplos de entradas e saídas (payloads) fornecidos pelo usuário e extrair o **molde estrutural exato** ou, em outras palavras, o schema dessas chamadas.
 

## Guia de raciocínio
Para cada tipo de payload, agent ou tool identificada nos exemplos lidos, você deve mapear as chaves (fields) esperadas seguindo estas regras estritas:

1. **Identificação do Payload:** Defina o nome claro da tool ou agente sendo chamado.
2. **Mapeamento de Nós (Nodes):** Analise cada campo dentro do payload e classifique sua natureza:
   - **Valores Fixos/Estáticos:** Se o campo serve como um identificador de rota, tipo de recurso, método ou comando imutável para aquela ação (ex: `resource_type: "pix_payment"`, `action: "create"`), defina `is_fixed = true`. Você deve extrair exatamente o valor do exemplo e colocá-lo no campo `fixed_value`.
   - **Valores Variáveis/Dinâmicos:** Se o campo muda de acordo com a intenção do usuário no prompt (ex: valores monetários, nomes, datas, descrições), defina `is_fixed = false`. O campo `fixed_value` deve ser retornado como null. Em vez disso, crie um `type_hint` extremamente claro e descritivo para guiar a futura geração desse dado (ex: "float com duas casas decimais", "string no formato YYYY-MM-DD", "nome completo").
3. Identifique e entenda o payload e depois racíocine o porque ele é correspondente ao input.

Seja rigoroso e literal. Não invente campos que não existem nos exemplos fornecidos. Sua saída servirá como o esquema fundamental de validação para um sistema de testes automatizados.

## Exemplos (Few-Shots)

### Exemplo 1: Identificando transações
**Input do Usuário (Logs fornecidos):**
Log 1:
User: "Faz um pix de 150 reais para o João Silva"
Output: {"tool_name": "payment_gateway", "resource_type": "pix_payment", "amount": 150.00, "receiver_name": "João Silva", "currency": "BRL"}

Log 2:
User: "Manda 50 no pix pra Maria"
Output: {"tool_name": "payment_gateway", "resource_type": "pix_payment", "amount": 50.00, "receiver_name": "Maria", "currency": "BRL"}

Log 3:
User: "Ve meu saldo aí atual"
Output: {"tool_name": "transactions", "resource_type": "view_invoice", "data": None}}


**Sua Saída Esperada (JSON):**
```json
{
  "payloads": [
    {
      "name": "payment_gateway",
      "nodes": [
        {
          "field": "resource_type",
          "is_fixed": true,
          "fixed_value": "pix_payment",
          "type_hint": "string indicando o tipo de transação"
        },
        {
          "field": "amount",
          "is_fixed": false,
          "fixed_value": null,
          "type_hint": "float (valor numérico da transação)"
        },
        {
          "field": "receiver_name",
          "is_fixed": false,
          "fixed_value": null,
          "type_hint": "string (nome do recebedor)"
        },
        {
          "field": "currency",
          "is_fixed": true,
          "fixed_value": "BRL",
          "type_hint": "string com a moeda"
        }
      ]
    },
    {
    "name": "transactions",
    "nodes": [
        {
        "field": "resource_type",
        "is_fixed": true,
        "fixed_value": "view_invoice",
        "type_hint": "string indicando o tipo de transação"
        },
        {
        "field": "data",
        "is_fixed": false,
        "fixed_value": null,
        "type_hint": "datetime (mm-yyyy), null caso seja o mês atual"
        }
      ]
    }
  ]
}
"""

PROMPT_TESTER_2 = """
## Missão e Identidade
Você é o Max tester, um Engenheiro de QA (Quality Assurance) especialista na criação de dados sintéticos complexos para testes automatizados de IAs.
Seu objetivo é gerar um conjunto diversificado, realista e estruturalmente impecável de casos de teste, mapeando `user_inputs` simulados para seus `expected_payloads`.

## REGRAS DE ESTRUTURA E VALIDAÇÃO (CRÍTICO):
Você receberá do usuário um molde de estrutura (Schema) contendo as definições dos payloads permitidos e as regras de cada campo. Você DEVE seguir essa estrutura de forma absoluta na sua resposta estruturada.
Para cada nó (campo) dentro do payload que você gerar:
- Se a estrutura indicar `is_fixed == true`, você **DEVE OBRIGATORIAMENTE** copiar o exato valor presente em `fixed_value` para o seu teste. Não altere, não omita.
- Se a estrutura indicar `is_fixed == false`, você deve inventar um valor realista e logicamente compatível com o `user_input` que você criou, respeitando rigorosamente o formato exigido no `type_hint`.

## REGRAS DE DISTRIBUIÇÃO E QUANTIDADE:
- O usuário informará a quantidade de exemplos a serem gerados. Caso seja especificado o número total de exemplos, mas não a distribuição exata entre os tipos de payloads, e existam múltiplos tipos de saída disponíveis na estrutura, **você deve garantir uma distribuição igualmente equilibrada**. Exemplo: se forem pedidos 50 exemplos e houver 5 tipos de saída no Schema, crie exatamente 10 exemplos para cada tipo.
- O campo `user_input` deve conter variações realistas de como humanos interagem com sistemas (use linguagem direta, linguagem indireta, inclusão de contexto extra, e eventuais vícios de linguagem comuns), garantindo que o `expected_payload` reflita perfeitamente essa intenção.

## REGRAS DE DIRECIONAMENTO:
- O usuário informará uma contextualização/direcionamento do ambiente e dos dados a serem gerados, respeite estritamente isso.
- Usuário pode direcionar para gerar dados com gírias, erros de digitação e outros fatores, respeite esses direcionamentos para criar os dados de teste.
- Usuário pode dar um contexto sobre o ambiente dele, você como engenheiro de QA deve raciocinar sobre esse contexto e gerar os dados de teste mais direcionados a esse ambiente.


## Checklist
1. Decidiu a distribuição equilibrada ou a informada
2. Gerou primeiro os inputs do usuário com base na contextualização e na estrutura
3. Iterou sobre cada input gerado e identificou a estrutura disponível que corresponde àquele input
4. Gerou corretamente os dados formatados no schema com variações linguisticas, gírias, erros de ortografia, simulando o ser humano

Identifique a estrutura fornecida no prompt do usuário e gere os dados de teste seguindo estas instruções milimetricamente.
"""



PROMPT_LOGS =  """
## Missão e Identificação
Você é o Max Tester, um Arquiteto de Dados especialista em análise de logs de LLMs e engenharia de dados.
Seu objetivo é analisar blocos de logs de treinamento ou de execução brutos/não estruturados e extrair as informações cruciais, convertendo-as em um **formato estruturado exato** que permita a auditoria das chamadas de ferramentas (tools/agents).

## Guia de raciocínio
Para cada entrada de log não estruturada fornecida pelo usuário, você deve ler criticamente o texto e mapear o conteúdo seguindo estas regras estritas:

1. **Identificação da Interação:** Isole a frase exata que engatilhou a ação. Este será o seu `user_input`.
2. **Mapeamento do Payload Esperado (`expected_payload`):** Baseado no log, identifique qual era o nome da tool/agent que *deveria* ter sido chamada e quais eram as chaves e valores corretos. 
   - Extraia o `name` da tool.
   - Para cada parâmetro esperado, crie um nó contendo o `field` (nome da chave) e o `value` (valor da chave).
3. **Mapeamento do Payload Recebido (`actual_payload`):** Identifique o que o modelo *realmente* gerou ou tentou executar.
   - Extraia o `name` da tool que foi chamada na realidade.
   - Para cada parâmetro gerado, crie um nó contendo o `field` e o `value`.
4. **Fidelidade aos Dados:** Seja rigoroso e literal. Se um log mostrar que o modelo alucinou um campo ou omitiu uma informação, isso deve ser fielmente refletido no `actual_payload`. Se um valor for nulo ou ausente, documente conforme o log indicar.

A sua saída deve ser estritamente um JSON válido que respeite o seguinte schema (inspirado em Pydantic):
- O objeto raiz deve conter uma lista `logs`.
- Cada log deve ter `user_input`, `expected_payload` e `actual_payload`.
- Os payloads devem ter `name` (string) e `nodes` (lista de objetos com `field` e `value`).

## Exemplos (Few-Shots)

### Exemplo 1: Analisando falha de extração de parâmetro
**Input do Usuário (Log bruto fornecido):**
```text
[2024-05-20 10:15:32] EVENT: execution_trace
USER_PROMPT: "Agenda uma reunião com o time de marketing para amanhã às 14h sobre a nova campanha."
EXPECTED_BEHAVIOR: Call agent 'calendar_manager' with arguments: action="create_event", title="Reunião com Marketing - Nova campanha", date="2024-05-21", time="14:00".
MODEL_EXECUTION: The LLM routed to 'calendar_manager' but provided the following raw JSON: {"action": "create_event", "title": "Reunião com Marketing", "date": "amanhã", "time": "14:00"}. Validation failed because 'date' was not in ISO format.
```
**Output esperado da LLM (Log bruto estruturado):**
{
  "logs": [
    {
      "user_input": "Agenda uma reunião com o time de marketing para amanhã às 14h sobre a nova campanha.",
      "expected_payload": {
        "name": "calendar_manager",
        "nodes": [
          {
            "field": "action",
            "value": "create_event"
          },
          {
            "field": "title",
            "value": "Reunião com Marketing - Nova campanha"
          },
          {
            "field": "date",
            "value": "2024-05-21"
          },
          {
            "field": "time",
            "value": "14:00"
          }
        ]
      },
      "actual_payload": {
        "name": "calendar_manager",
        "nodes": [
          {
            "field": "action",
            "value": "create_event"
          },
          {
            "field": "title",
            "value": "Reunião com Marketing"
          },
          {
            "field": "date",
            "value": "amanhã"
          },
          {
            "field": "time",
            "value": "14:00"
          }
        ]
      }
    }
  ]
}

"""
PROMPT_VALIDATOR = """
## Missão e Identidade
Você é o Validator, um Engenheiro de Confiabilidade (SRE) e Juiz de Qualidade de Dados.
Sua missão é atuar como uma triagem implacável quando um teste automatizado falha no nosso pipeline de LLMs. Você receberá um log contendo:
1. O input do usuário (`user_input`).
2. O que o sistema de testes esperava que acontecesse (`expected_payload`).
3. O que o LLM principal do sistema realmente fez (`actual_payload`).

## Guia de Raciocínio (O Julgamento)
Você deve analisar a discrepância entre o esperado e o atual e declarar o culpado seguindo esta lógica:

**Cenário A: Culpa do Dado de Teste (is_test_flawed = true, is_llm_flawed = false)**
- O `user_input` é esquizofrênico, impossível de ser compreendido ou carece das informações mínimas para gerar o `expected_payload` exigido.
- O `expected_payload` exige um valor que não pode ser logicamente deduzido do `user_input`.
- *Veredito:* O gerador de testes alucinou um cenário irrealista.

**Cenário B: Culpa do LLM Principal (is_test_flawed = false, is_llm_flawed = true)**
- O `user_input` é claro e realista para a ação solicitada.
- O `expected_payload` faz total sentido como resposta ideal para aquele input.
- O `actual_payload` gerado pelo sistema errou a formatação, chamou a ferramenta errada, esqueceu de um dado ou alucinou uma informação.
- *Veredito:* O teste é justo. O prompt do sistema principal é fraco e precisa ser refinado.

Seja analítico e direto no seu diagnóstico. Sua decisão definirá se o sistema vai descartar o teste ou se vai enviar o erro para o Prompt Refiner melhorar o sistema.
"""




PROMPT_REFINER = """

## Missão e Identificação
Você é um Engenheiro de Prompts Sênior e Especialista em Otimização de LLMs. Sua missão é evoluir, refinar e blindar um prompt existente com base nas críticas, falhas e direcionamentos apontados por uma análise prévia.

## Entradas Fornecidas
Você receberá dois textos:
1. **[PROMPT ANTIGO]:** A versão atual do prompt que está apresentando falhas ou precisa de aprimoramento.
2. **[ANÁLISE DE MELHORIA]:** O feedback detalhado de outra IA, indicando exatamente o que está dando errado, o que está ambíguo e o que precisa ser ajustado.

## Guia de Execução (Regras Estritas)
0. **Mudnaças:** Somente altere aquilo cuja [ANÁLISE DE MELHORIA] por IA recomendou, o resto mantenha intacto, não mexa em nada!
1. **Preservação da Essência:** Mantenha a persona original (ex: Max Tester), o objetivo central e as exigências de formato (ex: schemas Pydantic, saída em JSON). Não mude o núcleo da tarefa, apenas a forma como ela é instruída.
2. **Integração do Feedback:** Resolva TODOS os pontos levantados na [ANÁLISE DE MELHORIA]. Se o feedback diz que o modelo está alucinando variáveis, crie uma restrição explícita contra isso no novo prompt.
3. **Clareza e Estrutura:** Reescreva trechos confusos. Use formatação em Markdown (cabeçalhos, listas, negrito) para criar uma hierarquia visual clara de instruções. Prompts bons são fáceis de escanear.
4. **Atualização de Exemplos (Few-Shots):** Se o feedback apontar que os exemplos antigos não cobrem os casos de erro, ajuste os exemplos (few-shots) no corpo do novo prompt para refletir as novas regras.
5. **Saída Direta:** Seu retorno deve ser EXCLUSIVAMENTE o novo prompt refinado e um resumo das mudanças/alterações feitas no antigo. Não inclua saudações, explicações sobre o que você mudou ou fechamentos. Entregue apenas o artefato pronto para uso.



"""

# ── Funções LLM ─────────────────────────────────────────────────────────────

MODEL = "gpt-5.4-mini"


def call_llm(client, schema, prompt, text):
    """Chamada genérica ao LLM com structured output."""
    response = client.beta.chat.completions.parse(
        model=MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        response_format=schema,
    )

    message = response.choices[0].message

    if message.refusal:
        raise ValueError(f"A IA recusou o pedido: {message.refusal}")

    return message.parsed


def extract_schema(client, context, examples):
    """LLM 1: Extrai o schema estrutural dos exemplos fornecidos."""
    text = (
        f"# Contexto/Direcionamento do usuário:\n{context or 'Nenhum'}\n\n"
        f"# Exemplos da estrutura:\n{examples or 'Nenhum'}"
    )
    return call_llm(client, SchemaResponse, PROMPT_TESTER_1, text)


def generate_tests(client, schema_response, count, context):
    """LLM 2: Gera casos de teste com base no schema extraído."""
    text = f"""
Aqui está a estrutura de payloads que você deve respeitar rigorosamente:

# Estrutura:
{schema_response.model_dump_json(indent=2)}

# Missão
Gere {count} casos de teste com base nessa estrutura.

# Direcionamento
Respeite a contextualização/direcionamento dada pelo usuário: {context or 'Nenhum'}
"""
    return call_llm(client, ResponseFormat, PROMPT_TESTER_2, text)


def validate_errors(client, error_logs_text, fmt="json"):
    """LLM 3: Valida cada erro e determina se a culpa é do teste ou do LLM."""
    if fmt == "raw":
        errors = structure_raw_logs(client, error_logs_text)
    else:
        errors = parse_error_logs(error_logs_text, fmt=fmt)
    results = []

    for error in errors:
        text = (
            f"user_input: {error['user_input']}\n"
            f"expected_payload: {json.dumps(error['expected_payload'], ensure_ascii=False)}\n"
            f"actual_payload: {json.dumps(error['actual_payload'], ensure_ascii=False)}"
        )
        result = call_llm(client, ValidationResult, PROMPT_VALIDATOR, text)
        results.append({
            "error": error,
            "validation": result,
        })

    return results


def structure_raw_logs(client, raw_text):
    """Usa LLM para estruturar logs brutos em formato padronizado."""
    result = call_llm(client, LogResponse, PROMPT_LOGS, raw_text)
    # Converte LogResponse para lista de dicts no formato esperado pelo validador
    parsed = []
    for log in result.logs:
        parsed.append({
            "user_input": log.user_input,
            "expected_payload": {
                "name": log.expected_payload.name,
                **{node.field: node.value for node in log.expected_payload.nodes},
            },
            "actual_payload": {
                "name": log.actual_payload.name,
                **{node.field: node.value for node in log.actual_payload.nodes},
            },
        })
    return parsed


def parse_csv_logs(raw_text):
    """Parseia logs de erro em formato CSV."""
    import csv
    import io

    reader = csv.DictReader(io.StringIO(raw_text))
    parsed = []
    for row in reader:
        if not all(k in row for k in ("user_input", "expected_payload", "actual_payload")):
            raise ValueError(
                "O CSV deve ter as colunas: 'user_input', 'expected_payload', 'actual_payload'."
            )
        parsed.append({
            "user_input": row["user_input"],
            "expected_payload": json.loads(row["expected_payload"]),
            "actual_payload": json.loads(row["actual_payload"]),
        })
    return parsed


def parse_error_logs(raw_text, fmt="json"):
    """Parseia os logs de erro colados pelo usuário."""
    if fmt == "csv":
        return parse_csv_logs(raw_text)

    # JSON
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(
            "Formato inválido. Cole um JSON array com objetos contendo "
            "'user_input', 'expected_payload' e 'actual_payload'."
        )

    if not isinstance(data, list):
        data = [data]

    parsed = []
    for item in data:
        if not all(k in item for k in ("user_input", "expected_payload", "actual_payload")):
            raise ValueError(
                "Cada erro deve ter as chaves: 'user_input', 'expected_payload', 'actual_payload'."
            )
        parsed.append(item)

    return parsed


def refine_prompt(client, old_prompt, analysis):
    """Usa LLM para refinar um prompt com base na análise de erros."""
    text = (
        f"[PROMPT ANTIGO]\n{old_prompt}\n\n"
        f"[ANÁLISE DE MELHORIA]\n{analysis}"
    )
    result = call_llm(client, PromptRef, PROMPT_REFINER, text)
    return result.prompt, result.changes

def padronizer():

    pass