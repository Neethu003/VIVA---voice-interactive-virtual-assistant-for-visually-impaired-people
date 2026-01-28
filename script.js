const objectsList = document.getElementById("objects-list");
const toggleBtn = document.getElementById("toggle-detection");
const alertIndicator = document.getElementById("alert-indicator");
const languageSelect = document.getElementById("language-select");

let detectionEnabled = true;

// Toggle detection
toggleBtn.addEventListener("click", () => {
    fetch("/toggle_detection", { method: "POST" })
    .then(res => res.json())
    .then(data => {
        detectionEnabled = data.detection_enabled;
        toggleBtn.textContent = detectionEnabled ? "Detection ON" : "Detection OFF";
    });
});

// Change language
languageSelect.addEventListener("change", () => {
    const lang = languageSelect.value;
    fetch("/set_language", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ lang })
    });
});

// Fetch detected objects from backend
function updateObjects() {
    if(!detectionEnabled) return;

    fetch("/detections")
        .then(res => res.json())
        .then(data => {
            objectsList.innerHTML = '';
            let alertColor = "green";

            data.forEach(obj => {
                const div = document.createElement("div");
                div.classList.add("object-item");

                if(obj.dist <= 0.75) {
                    div.classList.add("close");
                    alertColor = "red";
                } else if(obj.dist <= 1.5) {
                    div.classList.add("near");
                    if(alertColor !== "red") alertColor = "orange";
                }

                div.textContent = `${obj.name} | ${obj.dist}m | ${obj.direction}`;
                objectsList.appendChild(div);
            });

            alertIndicator.style.backgroundColor = alertColor;
        });
}

setInterval(updateObjects, 1500);
