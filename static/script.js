const form = document.getElementById("fraudForm");
const resultBox = document.getElementById("resultBox");
const resultText = document.getElementById("resultText");
const riskLevel = document.getElementById("riskLevel");
const fraudProb = document.getElementById("fraudProb");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const data = {
    trans_datetime: document.getElementById("trans_datetime").value,
    category: Number(document.getElementById("category").value),
    dob: document.getElementById("dob").value,
    trans_amount: Number(document.getElementById("trans_amount").value),
    state: Number(document.getElementById("state").value),
    zip: Number(document.getElementById("zip").value),
  };

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const result = await response.json();

    resultBox.classList.remove("hidden");
    resultBox.classList.remove("low", "medium", "high");

    if (!response.ok) {
      // Show backend error message if prediction failed
      resultText.innerText = "Error";
      riskLevel.innerText = "-";
      fraudProb.innerText = result.detail || "Request failed";
      return;
    }

    fraudProb.innerText = result.fraud_probability.toFixed(3);
    riskLevel.innerText = result.risk_level;
    resultText.innerText = result.result;

    if (result.risk_level === "LOW_RISK") {
      resultBox.classList.add("low");
    } else if (result.risk_level === "MEDIUM_RISK") {
      resultBox.classList.add("medium");
    } else {
      resultBox.classList.add("high");
    }
  } catch (err) {
    console.error(err);
    resultBox.classList.remove("hidden");
    resultText.innerText = "Error";
    riskLevel.innerText = "-";
    fraudProb.innerText = "Could not reach API";
  }
});
