![Trade Detection Demo](./docs/assets/simple_trade.png)

### Example Output
When a trade image is processed, the engine returns a structured JSON object:

\`\`\`json
{
    "outgoing": {
        "item_count": 4,
        "robux_value": 0,
        "items": [
            {
                "id": 1031429,
                "name": "Domino Crown"
            },
            {
                "id": 72082328,
                "name": "Red Sparkle Time Fedora"
            },
            {
                "id": 124730194,
                "name": "Blackvalk"
            },
            {
                "id": 16652251,
                "name": "Red Tango"
            }
        ]
    },
    "incoming": {
        "item_count": 2,
        "robux_value": 1048576,
        "items": [
            {
                "id": 21070012,
                "name": "Dominus Empyreus"
            },
            {
                "id": 22850569,
                "name": "Red Bandana of SQL Injection"
            }
        ]
    }
}
\`\`\`

| Input Detection | Resolved Data (JSON) |
| :--- | :--- |
| ![Detection](./docs/assets/simple_trade.png) | \`\`\`json
{
    "outgoing": {
        "item_count": 4,
        "robux_value": 0,
        "items": [
            {
                "id": 1031429,
                "name": "Domino Crown"
            },
            {
                "id": 72082328,
                "name": "Red Sparkle Time Fedora"
            },
            {
                "id": 124730194,
                "name": "Blackvalk"
            },
            {
                "id": 16652251,
                "name": "Red Tango"
            }
        ]
    },
    "incoming": {
        "item_count": 2,
        "robux_value": 1048576,
        "items": [
            {
                "id": 21070012,
                "name": "Dominus Empyreus"
            },
            {
                "id": 22850569,
                "name": "Red Bandana of SQL Injection"
            }
        ]
    }
}
\`\`\` |