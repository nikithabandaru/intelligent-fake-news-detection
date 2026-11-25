document.addEventListener('DOMContentLoaded', () => {
    const newsInput = document.getElementById('newsInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    const predictionLabel = document.getElementById('predictionLabel');
    const confidenceValue = document.getElementById('confidenceValue');
    const errorMsg = document.getElementById('errorMsg');
    const shortWarning = document.getElementById('shortWarning');

    function setLoading(isLoading) {
        loadingDiv.classList.toggle('hidden', !isLoading);
        analyzeBtn.disabled = isLoading;
        clearBtn.disabled = isLoading;
    }

    function clearResult() {
        resultDiv.classList.add('hidden');
        errorMsg.classList.add('hidden');
        predictionLabel.textContent = '';
        confidenceValue.textContent = '';
        shortWarning.textContent = '';
        shortWarning.classList.add('hidden');
    }

    analyzeBtn.addEventListener('click', async () => {
        clearResult();
        const text = newsInput.value.trim();
        if (!text) {
            errorMsg.textContent = "Please enter some news text to analyze.";
            errorMsg.classList.remove('hidden');
            return;
        }
        // Detect very short input (less than 30 characters or a single short sentence)
        if (text.length < 30 || (!text.includes('.') && text.split(' ').length <= 5)) {
            shortWarning.textContent = "⚠️ Warning: Very short claims may produce less reliable predictions. The model detects learned text patterns and does not independently verify facts.";
            shortWarning.classList.remove('hidden');
        }
        setLoading(true);
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ news_text: text })
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Server error');
            }
            // Show result
            predictionLabel.textContent = data.prediction;
            predictionLabel.className = data.prediction === 'FAKE' ? 'fake' : 'real';
            confidenceValue.textContent = (data.confidence !== null ? (data.confidence * 100).toFixed(2) : 'N/A');
            resultDiv.classList.remove('hidden');
        } catch (err) {
            errorMsg.textContent = err.message;
            errorMsg.classList.remove('hidden');
        } finally {
            setLoading(false);
        }
    });

    clearBtn.addEventListener('click', () => {
        newsInput.value = '';
        clearResult();
    });
});