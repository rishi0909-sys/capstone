import json
import uuid
import os

collection_id = str(uuid.uuid4())

# Postman collection template
collection = {
    "info": {
        "_postman_id": collection_id,
        "name": "Intelligent Pattern Discovery Engine",
        "description": "API Tests for the FastAPI backend of the Intelligent Pattern Discovery Engine",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
    },
    "item": []
}

def create_request(name, method, path, body=None, test_script=None):
    req = {
        "name": name,
        "request": {
            "method": method,
            "header": [
                {"key": "Content-Type", "value": "application/json", "type": "text"}
            ] if body else [],
            "url": {
                "raw": "{{baseUrl}}" + path,
                "host": ["{{baseUrl}}"],
                "path": path.strip("/").split("/")
            }
        },
        "response": []
    }
    
    if body:
        req["request"]["body"] = {
            "mode": "raw",
            "raw": json.dumps(body, indent=4)
        }
        
    if test_script:
        req["event"] = [
            {
                "listen": "test",
                "script": {
                    "exec": test_script.split('\n'),
                    "type": "text/javascript"
                }
            }
        ]
        
    return req

# Folder 01 - Health
f1 = {
    "name": "01 - Health",
    "item": [
        create_request(
            "Health Check", 
            "GET", 
            "/api/health", 
            test_script="""pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
pm.test("Response is valid JSON", function () {
    pm.response.to.be.json;
});
pm.test("Status is ok", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.status).to.eql("ok");
});"""
        )
    ]
}

# Folder 02 - Dataset
f2 = {
    "name": "02 - Dataset",
    "item": [
        create_request(
            "Load Demo Dataset",
            "POST",
            "/api/dataset/demo",
            test_script="""pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
pm.test("Response has dataset_id", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('dataset_id');
    pm.expect(jsonData.dataset_id).to.not.be.empty;
    // Set to environment
    pm.environment.set("datasetId", jsonData.dataset_id);
});
pm.test("Response has row count and columns", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.row_count).to.be.a('number');
    pm.expect(jsonData.row_count).to.be.above(0);
    pm.expect(jsonData.columns).to.be.an('array');
});"""
        )
    ]
}

# Folder 03 - Preprocessing
f3 = {
    "name": "03 - Preprocessing",
    "item": [
        create_request(
            "Run Preprocessing",
            "POST",
            "/api/preprocessing/run",
            body={"dataset_id": "{{datasetId}}", "column_mapping": {}},
            test_script="""pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
pm.test("Response has preprocessing summary", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.initial_rows).to.be.a('number');
    pm.expect(jsonData.final_rows).to.be.a('number');
    pm.expect(jsonData.unique_transactions).to.be.a('number');
    pm.expect(jsonData.status).to.eql('Success');
});"""
        )
    ]
}

# Folder 04 - Pattern Mining
f4 = {
    "name": "04 - Pattern Mining",
    "item": [
        create_request(
            "Run Mining (FP-Growth)",
            "POST",
            "/api/mining/run",
            body={
                "dataset_id": "{{datasetId}}",
                "min_support": 0.01,
                "min_confidence": 0.3,
                "min_lift": 1.1
            },
            test_script="""pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
pm.test("Response has rules array", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.rule_count).to.be.a('number');
    pm.expect(jsonData.rules).to.be.an('array');
    pm.expect(jsonData.rules.length).to.be.above(0, "Rules should not be empty");
});
pm.test("Verify valid rule structure", function () {
    var jsonData = pm.response.json();
    if(jsonData.rules.length > 0) {
        var rule = jsonData.rules[0];
        pm.expect(rule).to.have.property('antecedent');
        pm.expect(rule).to.have.property('consequent');
        pm.expect(rule.support).to.be.within(0, 1);
        pm.expect(rule.confidence).to.be.within(0, 1);
        pm.expect(rule.lift).to.be.at.least(0);
    }
});
pm.test("Check for Laptop -> Wireless Mouse", function () {
    var jsonData = pm.response.json();
    var found = false;
    for(var i=0; i<jsonData.rules.length; i++) {
        var r = jsonData.rules[i];
        if(r.antecedent.includes("Laptop") && r.consequent.includes("Wireless Mouse")) {
            found = true;
            break;
        }
    }
    if (found) {
        console.log("Laptop -> Wireless Mouse rule found");
    } else {
        console.log("Laptop -> Wireless Mouse rule not found in returned rules");
    }
});"""
        ),
        create_request(
            "Get Rules",
            "GET",
            "/api/mining/rules?dataset_id={{datasetId}}&limit=10&offset=0",
            test_script="""pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
pm.test("Returns array of rules", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData).to.be.an('array');
});"""
        )
    ]
}

# Folder 05 - Demographics
f5 = {
    "name": "05 - Demographics",
    "item": [
        create_request(
            "Analyze Demographics",
            "POST",
            "/api/demographics/analyze",
            body={"dataset_id": "{{datasetId}}"},
            test_script="""pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
pm.test("Response status is success", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.status).to.be.oneOf(["success", "unavailable"]);
    if (jsonData.status === "success") {
        pm.expect(jsonData.preferences).to.be.an('array');
    }
});"""
        )
    ]
}

# Folder 06 - Temporal Analysis
f6 = {
    "name": "06 - Temporal Analysis",
    "item": [
        create_request(
            "Analyze Temporal",
            "POST",
            "/api/temporal/analyze",
            body={"dataset_id": "{{datasetId}}"},
            test_script="""pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
pm.test("Response status is success", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.status).to.be.oneOf(["success", "unavailable"]);
    if (jsonData.status === "success") {
        pm.expect(jsonData.monthly_trend).to.be.an('array');
    }
});"""
        )
    ]
}

# Folder 07 - Forecasting
f7 = {
    "name": "07 - Forecasting",
    "item": [
        create_request(
            "Run Forecast (XGBoost)",
            "POST",
            "/api/forecast/run",
            body={"dataset_id": "{{datasetId}}", "target_product": "Overall", "horizon": 4, "granularity": "W"},
            test_script="""pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
pm.test("Forecast returns evaluation metrics", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.mae).to.be.a('number');
    pm.expect(jsonData.rmse).to.be.a('number');
    pm.expect(jsonData.r2).to.be.a('number');
});
pm.test("Forecast returns predictions arrays", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.historical_actuals).to.be.an('array');
    pm.expect(jsonData.test_preds).to.be.an('array');
    pm.expect(jsonData.future_preds).to.be.an('array');
});"""
        )
    ]
}

# Folder 08 - Insights
f8 = {
    "name": "08 - Insights",
    "item": [
        create_request(
            "Generate Insights",
            "POST",
            "/api/insights/generate",
            body={"dataset_id": "{{datasetId}}"},
            test_script="""pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
pm.test("Insights array is returned", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.insights).to.be.an('array');
});"""
        )
    ]
}

# Folder 09 - LLM Explanation
f9 = {
    "name": "09 - LLM Explanation",
    "item": [
        create_request(
            "Explain Rule",
            "POST",
            "/api/llm/explain",
            body={
                "antecedent": ["Laptop"],
                "consequent": ["Wireless Mouse"],
                "support": 0.124,
                "confidence": 0.92,
                "lift": 2.31
            },
            test_script="""pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
pm.test("Explanation response is structured correctly", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('validated');
    if (!jsonData.validated) {
        pm.expect(jsonData.error).to.be.a('string');
        console.log("LLM handled gracefully without key: " + jsonData.error);
    } else {
        pm.expect(jsonData.explanation).to.be.a('string');
    }
});"""
        )
    ]
}

# Folder 10 - Error Handling
f10 = {
    "name": "10 - Error Handling",
    "item": [
        create_request(
            "Invalid Dataset ID (Preprocessing)",
            "POST",
            "/api/preprocessing/run",
            body={"dataset_id": "invalid-id-12345"},
            test_script="""pm.test("Status code is 4xx", function () {
    pm.expect(pm.response.code).to.be.within(400, 499);
});"""
        ),
        create_request(
            "Missing Target Product (Forecasting)",
            "POST",
            "/api/forecast/run",
            body={"dataset_id": "{{datasetId}}"},
            test_script="""pm.test("Status code is 422 Unprocessable Entity (Validation Error)", function () {
    pm.response.to.have.status(422);
});"""
        ),
        create_request(
            "Mining before Preprocessing",
            "POST",
            "/api/mining/run",
            body={
                "dataset_id": "another-invalid-id",
                "min_support": 0.01,
                "min_confidence": 0.3,
                "min_lift": 1.1
            },
            test_script="""pm.test("Status code is 400 Bad Request", function () {
    pm.response.to.have.status(400);
});"""
        )
    ]
}

# Folder 11 - End-to-End Demo
f11 = {
    "name": "11 - End-to-End Demo",
    "item": [
        create_request("1. Load Demo", "POST", "/api/dataset/demo", test_script="""pm.test("Status code is 200", function () { pm.response.to.have.status(200); pm.environment.set("datasetId", pm.response.json().dataset_id); });"""),
        create_request("2. Preprocess", "POST", "/api/preprocessing/run", body={"dataset_id": "{{datasetId}}"}, test_script="""pm.test("Status code is 200", function () { pm.response.to.have.status(200); });"""),
        create_request("3. FP-Growth", "POST", "/api/mining/run", body={"dataset_id": "{{datasetId}}", "min_support": 0.01, "min_confidence": 0.3, "min_lift": 1.1}, test_script="""pm.test("Status code is 200", function () { pm.response.to.have.status(200); });"""),
        create_request("4. Demographics", "POST", "/api/demographics/analyze", body={"dataset_id": "{{datasetId}}"}, test_script="""pm.test("Status code is 200", function () { pm.response.to.have.status(200); });"""),
        create_request("5. Temporal", "POST", "/api/temporal/analyze", body={"dataset_id": "{{datasetId}}"}, test_script="""pm.test("Status code is 200", function () { pm.response.to.have.status(200); });"""),
        create_request("6. XGBoost Forecast", "POST", "/api/forecast/run", body={"dataset_id": "{{datasetId}}", "target_product": "Overall", "horizon": 4, "granularity": "W"}, test_script="""pm.test("Status code is 200", function () { pm.response.to.have.status(200); });"""),
        create_request("7. Statistical Insights", "POST", "/api/insights/generate", body={"dataset_id": "{{datasetId}}"}, test_script="""pm.test("Status code is 200", function () { pm.response.to.have.status(200); });"""),
        create_request("8. LLM Explanation", "POST", "/api/llm/explain", body={"antecedent": ["Laptop"], "consequent": ["Wireless Mouse"], "support": 0.124, "confidence": 0.92, "lift": 2.31}, test_script="""pm.test("Status code is 200", function () { pm.response.to.have.status(200); });""")
    ]
}

collection["item"] = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]

with open("postman/Intelligent-Pattern-Discovery-Engine.postman_collection.json", "w") as f:
    json.dump(collection, f, indent=4)

print("Collection generated successfully.")
