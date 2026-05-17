/* static/script.js */

document.addEventListener("DOMContentLoaded", () => {
    // 1. Elements Retrieval
    const hinglishInput = document.getElementById("hinglishInput");
    const charCount = document.getElementById("charCount");
    const transliterateBtn = document.getElementById("transliterateBtn");
    
    // Status indicators
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");
    const statusSeq2Seq = document.getElementById("statusSeq2Seq");
    const statusAttention = document.getElementById("statusAttention");
    
    // Output slots
    const outputRule = document.getElementById("outputRule");
    const outputSeq2Seq = document.getElementById("outputSeq2Seq");
    const outputAttention = document.getElementById("outputAttention");
    
    // Preset triggers
    const presetButtons = document.querySelectorAll(".preset-btn");

    // 2. Character Counter Event
    hinglishInput.addEventListener("input", () => {
        const count = hinglishInput.value.length;
        charCount.textContent = `${count} / 250 characters`;
    });

    // 3. Preset Clicks Event
    presetButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const presetText = btn.getAttribute("data-preset");
            hinglishInput.value = presetText;
            charCount.textContent = `${presetText.length} / 250 characters`;
            
            // Auto run transliteration on preset selection
            runTransliteration(presetText);
        });
    });

    // 4. Manual Transliterate Click Event
    transliterateBtn.addEventListener("click", () => {
        const text = hinglishInput.value.trim();
        if (!text) {
            alert("Please type some Hinglish text to evaluate!");
            return;
        }
        runTransliteration(text);
    });

    // 5. Query Model Deployment Status at Startup
    checkNeuralWeightsStatus();

    // Helper: Core Transliteration REST query
    function runTransliteration(text) {
        // Render typing loading state in cards
        const loaderHTML = `
            <div class="typing-loader">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        `;
        
        outputRule.innerHTML = loaderHTML;
        outputSeq2Seq.innerHTML = loaderHTML;
        outputAttention.innerHTML = loaderHTML;
        
        fetch("/compare", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: text })
        })
        .then(response => {
            if (!response.ok) throw new Error("Server comparison error");
            return response.json();
        })
        .then(data => {
            // Write predictions back to cards with clean animations
            outputRule.innerHTML = `<p class="fade-in">${data.rule}</p>`;
            outputSeq2Seq.innerHTML = `<p class="fade-in">${data.seq2seq}</p>`;
            outputAttention.innerHTML = `<p class="fade-in">${data.attention}</p>`;
            
            // Double check weights status on run to keep dashboard updated
            updateIndividualCardsStatus(data.seq2seq_trained, data.attention_trained);
        })
        .catch(err => {
            console.error(err);
            const errHTML = `<p class="placeholder-text" style="color: hsl(38, 95%, 55%);">Failed to fetch output</p>`;
            outputRule.innerHTML = errHTML;
            outputSeq2Seq.innerHTML = errHTML;
            outputAttention.innerHTML = errHTML;
        });
    }

    // Helper: Checks backend startup loaded weights
    function checkNeuralWeightsStatus() {
        fetch("/status")
        .then(response => response.json())
        .then(data => {
            const sTrained = data.seq2seq_trained;
            const aTrained = data.attention_trained;
            
            // Update master status dot in header
            if (sTrained && aTrained) {
                statusDot.className = "status-dot online";
                statusText.textContent = "Neural Engines Engaged (CUDA/MPS/CPU)";
            } else if (sTrained || aTrained) {
                statusDot.className = "status-dot pulsing";
                statusText.textContent = "Hybrid Neural Mode Enabled";
            } else {
                statusDot.className = "status-dot pulsing";
                statusText.textContent = "Neural Models Training (Rule-Based Stub Active)";
            }
            
            updateIndividualCardsStatus(sTrained, aTrained);
        })
        .catch(err => {
            console.error("Failed to query model weights deployment status:", err);
            statusText.textContent = "Status Check Failed";
        });
    }

    // Helper: Updates card footer badges
    function updateIndividualCardsStatus(seqTrained, attnTrained) {
        if (seqTrained) {
            statusSeq2Seq.className = "stat-val status-trained";
            statusSeq2Seq.innerHTML = `<i class="fa-solid fa-circle-check"></i> Trained`;
        } else {
            statusSeq2Seq.className = "stat-val status-untrained";
            statusSeq2Seq.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Training`;
        }
        
        if (attnTrained) {
            statusAttention.className = "stat-val status-trained";
            statusAttention.innerHTML = `<i class="fa-solid fa-circle-check"></i> Trained`;
        } else {
            statusAttention.className = "stat-val status-untrained";
            statusAttention.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Training`;
        }
    }
});
