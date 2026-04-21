# Cedar Authorization Policies for I³

[Cedar](https://www.cedarpolicy.com/) is AWS's open-source authorization
policy language. It was designed for *provable* authorization — the Cedar
team uses SMT-based differential testing and has formally verified the
analyzer (see _"Cedar: A New Language for Expressive, Fast, Safe, and
Analyzable Authorization"_, Cutler et al., OOPSLA 2024). Cedar powers
Amazon Verified Permissions and is written in Rust.

## Why Cedar here?

The Kyverno + OPA layer covers *infrastructure* authorization (which pod
may run). Cedar covers *application* authorization (which end-user may
read whose diary). Its built-in schema validation and evaluator
determinism make it a better fit than ad-hoc Python checks for a
privacy-sensitive system like I³.

## Files

| File                          | Purpose                                    |
| ----------------------------- | ------------------------------------------ |
| `i3.cedar`                    | policy set modelling the I³ authz rules    |
| `schema.cedarschema.json`     | Cedar schema (entity types, actions)       |
| `tests/test_cases.json`       | 20+ request/decision expectations          |

## Running tests

```bash
# Install the Cedar CLI (Rust): cargo install cedar-policy-cli
cedar validate --schema schema.cedarschema.json --policies i3.cedar
cedar authorize --schema schema.cedarschema.json --policies i3.cedar \
  --request-json tests/test_cases.json
```

The Python adapter `i3/authz/cedar_adapter.py` wraps
[`cedarpy`](https://pypi.org/project/cedarpy/) and is exercised by
`tests/test_cedar_authz.py`.

## Formal-methods note

Cedar ships with a differential test harness (policy evaluator cross-
checked against a Lean proof-of-concept) and a static analyzer that can
answer questions such as "does policy P never grant `delete` to a
non-admin?" — see the Cedar paper, §5.

## References

- Cedar site: <https://www.cedarpolicy.com/>
- Cedar GitHub: <https://github.com/cedar-policy/cedar>
- Cutler et al., *"Cedar: A New Language for Expressive, Fast, Safe, and
  Analyzable Authorization"*, OOPSLA 2024.
- AWS Verified Permissions announcement (2023):
  <https://aws.amazon.com/verified-permissions/>
