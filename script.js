// script.js

// --- Global State & Configuration ---
const SECTIONS = ['home-section', 'diagnosis-form-section', 'results-section', 'analytics-section', 'contact-section'];
const VALIDATION_ERROR_MSG = 'Invalid value! Please enter a realistic agricultural value.';
const THRESHOLD_PERCENTAGE = 0.15; // 15% for Species Consistency Check

// Mapping categorical inputs to numerical for SVM training/prediction
const CATEGORICAL_MAPPING = {
    'stage': { 'Seedling': 0, 'Vegetative': 1, 'Budding': 2, 'Flowering': 3 },
    'fertilizerUsed': { 'No': 0, 'Yes': 1 },
    'leafColor': { 'Pale': 0, 'Yellow': 1, 'Normal': 2, 'Dark Green': 3 },
    'wiltingSigns': { 'No': 0, 'Yes': 1 },
};

// --- SVM MODEL IMPLEMENTATION ---
class LinearSVM {
    constructor(learningRate = 0.01, lambdaParam = 0.01, nIterations = 1000) {
        this.lr = learningRate;
        this.lambda = lambdaParam;
        this.nIterations = nIterations;
        this.weights = null; 
        this.bias = null;    
    }

    dot(x, w) {
        return x.reduce((acc, val, i) => acc + val * w[i], 0);
    }

    train(X, y) {
        const nFeatures = X[0].length;
        const nSamples = X.length;

        this.weights = new Array(nFeatures).fill(0);
        this.bias = 0;

        for (let i = 0; i < this.nIterations; i++) {
            const randIndex = Math.floor(Math.random() * nSamples);
            const x_i = X[randIndex];
            const y_i = y[randIndex];

            const condition = y_i * (this.dot(x_i, this.weights) + this.bias);

            if (condition >= 1) {
                this.weights = this.weights.map(w => w - this.lr * (2 * this.lambda * w));
            } else {
                this.weights = this.weights.map((w, j) => w - this.lr * (2 * this.lambda * w - y_i * x_i[j]));
                this.bias = this.bias - this.lr * (-y_i);
            }
        }
    }

    predict(X) {
        return this.dot(X, this.weights) + this.bias >= 0 ? 1 : -1;
    }
}

class OneVsRestClassifier {
    constructor(modelClass, classes) {
        this.modelClass = modelClass;
        this.classes = classes;
        this.classifiers = {};
    }

    train(X, y) {
        this.classes.forEach(cls => {
            const classifier = new this.modelClass();
            const y_transformed = y.map(label => label === cls ? 1 : -1);
            classifier.train(X, y_transformed);
            this.classifiers[cls] = classifier;
        });
    }

    predict(X) {
        let maxScore = -Infinity;
        let predictedClass = null;

        this.classes.forEach(cls => {
            const classifier = this.classifiers[cls];
            const score = classifier.dot(X, classifier.weights) + classifier.bias;

            if (score > maxScore) {
                maxScore = score;
                predictedClass = cls;
            }
        });

        return predictedClass;
    }
}

// Global variable to store trained models
const SVM_MODELS = {};

// --- Helper Functions for Data Preparation ---

const normalize = (value, min, max) => (value - min) / (max - min);

function prepareTrainingData(dataset, targetLabel, classes) {
    // 1. Feature Selection (10 features)
    const features = dataset.map(item => [
        CATEGORICAL_MAPPING.stage[item.stage],
        normalize(item.moisture, 5, 100),
        normalize(item.pH, 4.0, 9.0),
        normalize(item.light, 1, 12), // Normalized Max Light Exposure to 12
        CATEGORICAL_MAPPING.fertilizerUsed[item.fertilizer === 1 ? 'Yes' : 'No'],
        CATEGORICAL_MAPPING.leafColor[item.leafColor],
        CATEGORICAL_MAPPING.wiltingSigns[item.wilting === 1 ? 'Yes' : 'No'],
        normalize(item.flowerCount, 0, 100), // Max flowercount for normalization
        normalize(item.height, 1, 300), // Max height for normalization
        item.pestScore, // 0-4
    ]);

    // 2. Labels
    const labels = dataset.map(item => item[targetLabel]);

    // 3. Train the One-vs-Rest classifier
    const classifier = new OneVsRestClassifier(LinearSVM, classes);
    classifier.train(features, labels);

    return classifier;
}

function preprocessUserInput(formData) {
    // Aggregate the four binary pest inputs into a single numerical score (0-4)
    const pestScore = (formData.whitePowder === 'Yes' ? 1 : 0) +
                      (formData.holesInLeaves === 'Yes' ? 1 : 0) +
                      (formData.stickyLeaves === 'Yes' ? 1 : 0) +
                      (formData.insectVisibility !== 'None' ? 1 : 0);

    // Create the feature vector (X) matching the order of training features (10 features)
    const featureVector = [
        CATEGORICAL_MAPPING.stage[formData.stage],
        normalize(parseFloat(formData.moisture), 5, 100),
        normalize(parseFloat(formData.ph), 4.0, 9.0),
        normalize(parseFloat(formData.light), 1, 12),
        CATEGORICAL_MAPPING.fertilizerUsed[formData.fertilizerUsed],
        CATEGORICAL_MAPPING.leafColor[formData.leafColor],
        CATEGORICAL_MAPPING.wiltingSigns[formData.wiltingSigns],
        normalize(parseInt(formData.flowerCount), 0, 100),
        normalize(parseInt(formData.plantHeight), 1, 300),
        pestScore // Use the aggregated pest score
    ];

    return { featureVector, pestScore };
}


// --- Main SVM Training Function ---
function trainAllModels() {
    console.log("Starting SVM Model Training...");

    // Health Status: Healthy, Moderate, Unhealthy
    SVM_MODELS['HealthStatus'] = prepareTrainingData(FLOWER_DATASET, 'HealthStatus', ['Healthy', 'Moderate', 'Unhealthy']);

    // Growth Type: Slow, Normal, Fast
    SVM_MODELS['GrowthType'] = prepareTrainingData(FLOWER_DATASET, 'GrowthType', ['Slow', 'Normal', 'Fast']);

    // Risk Level: Low, Medium, High
    SVM_MODELS['RiskLevel'] = prepareTrainingData(FLOWER_DATASET, 'RiskLevel', ['Low', 'Medium', 'High']);

    console.log("SVM Models Trained Successfully.");
}


// --- Form Validation Logic ---
function validateInput(input, min, max, type) {
    const value = input.value.trim();
    const errorEl = document.getElementById(`${input.id}-error`);

    let isValid = true;
    let numValue;

    if (input.tagName === 'SELECT' && !value) {
        isValid = false;
    } else if (input.type === 'radio' && !document.querySelector(`input[name="${input.name}"]:checked`)) {
        // Validation for radio groups is complex, better to handle via required attribute on form submission
    } else if (type === 'number') {
        numValue = parseFloat(value);
        const numMin = parseFloat(min);
        const numMax = parseFloat(max);

        if (value === '' || isNaN(numValue) || numValue < numMin || numValue > numMax) {
            isValid = false;
        }
    }

    // Only show error for input/select fields here
    if (errorEl) {
        errorEl.classList.toggle('hidden', isValid);
        if (!isValid) {
            errorEl.textContent = VALIDATION_ERROR_MSG;
            input.setCustomValidity(VALIDATION_ERROR_MSG);
        } else {
            input.setCustomValidity('');
        }
    }
    return isValid;
}

function validateAllInputs(form) {
    let allValid = true;

    // Validate number inputs with min/max
    const numberInputs = [
        { id: 'moisture', min: 5, max: 100 },
        { id: 'ph', min: 4.0, max: 9.0 },
        { id: 'light', min: 1, max: 24 },
        { id: 'flowerCount', min: 0, max: 1000 },
        { id: 'plantHeight', min: 1, max: 500 }
    ];

    numberInputs.forEach(item => {
        const input = document.getElementById(item.id);
        if (input && !validateInput(input, item.min, item.max, 'number')) {
            allValid = false;
        }
    });
    
    // Validate select fields
    const selectInputs = document.querySelectorAll('#diagnosis-form select[required]');
    selectInputs.forEach(select => {
        if (!validateInput(select, null, null, 'select')) {
            allValid = false;
        }
    });

    // Validate radio groups (Check if AT LEAST one is checked)
    const radioGroups = ['fertilizerUsed', 'wiltingSigns', 'whitePowder', 'holesInLeaves', 'stickyLeaves'];
    radioGroups.forEach(groupName => {
        const checked = document.querySelector(`input[name="${groupName}"]:checked`);
        if (!checked) {
            allValid = false;
            // You can add a visual error indicator here if needed
        }
    });

    return allValid;
}


// --- Species Consistency Check ---
function runSpeciesConsistency(formData) {
    const species = formData.species;
    const userMoisture = parseFloat(formData.moisture);
    const userpH = parseFloat(formData.ph);
    const userLight = parseFloat(formData.light);
    const userHeight = parseInt(formData.plantHeight); 

    const ideal = SPECIES_IDEALS[species];
    const details = [];
    let isConsistent = true;

    const checkParameter = (userVal, idealParam, label) => {
        const idealVal = idealParam.ideal;
        const hardBoundaryCheck = userVal >= idealParam.min && userVal <= idealParam.max;

        if (!hardBoundaryCheck) {
            isConsistent = false;
        }

        const status = hardBoundaryCheck ? '✅ Consistent' : '⚠️ Warning';
        details.push({
            label,
            user: `${userVal} ${idealParam.unit}`,
            ideal: `${idealParam.ideal} ${idealParam.unit}`,
            range: `(${idealParam.min} - ${idealParam.max} ${idealParam.unit})`,
            status
        });
    };

    checkParameter(userMoisture, ideal.moisture, 'Soil Moisture');
    checkParameter(userpH, ideal.pH, 'Soil pH');
    checkParameter(userLight, ideal.light, 'Light Exposure');
    checkParameter(userHeight, ideal.height, 'Plant Height');

    return { isConsistent, details };
}

// --- Risk Decision Engine & Recommendations ---
function runDecisionEngine(svmResults, formData) {
    const recommendations = [];
    const highRisk = svmResults.RiskLevel === 'High';
    const slowGrowth = svmResults.GrowthType === 'Slow';
    
    // 1. IMMEDIATE ATTENTION REQUIRED (High Risk)
    if (highRisk) {
        recommendations.push({
            type: 'rec-high-risk',
            text: '🚨 IMMEDIATE ATTENTION REQUIRED: The model predicts a High Risk Level. Start checking for environmental extremes and pest visibility immediately.'
        });
    }

    // 2. Pest Control Recommendation
    const pestIndicators = (formData.whitePowder === 'Yes' || formData.holesInLeaves === 'Yes' || formData.stickyLeaves === 'Yes' || formData.insectVisibility !== 'None');
    if (pestIndicators) {
        recommendations.push({
            type: 'rec-pest',
            text: `🕷️ Pest Control Recommended: Indicators suggest an infestation. Use a targeted organic or chemical pesticide suitable for the detected pest type.`
        });
    }

    // 3. Watering Adjustment (Moisture)
    const idealMoisture = SPECIES_IDEALS[formData.species].moisture.ideal;
    const userMoisture = parseFloat(formData.moisture);
    
    if (userMoisture < (idealMoisture * 0.80)) {
        recommendations.push({
            type: 'rec-water',
            text: '💧 Watering Adjustment: Soil moisture is significantly low. Recommend increasing watering frequency to prevent wilting.'
        });
    } else if (userMoisture > (idealMoisture * 1.20)) {
        recommendations.push({
            type: 'rec-water',
            text: '🛑 Watering Adjustment: Soil moisture is too high. Ensure proper drainage to avoid root rot.'
        });
    }

    // 4. Soil Correction (pH Mismatch)
    const idealpH = SPECIES_IDEALS[formData.species].pH.ideal;
    const userpH = parseFloat(formData.ph);
    
    if (userpH < (idealpH - 0.7)) {
        recommendations.push({
            type: 'rec-soil',
            text: '🧪 Soil Correction (Low pH): Your soil is too acidic. Add lime or wood ash to raise the pH.'
        });
    } else if (userpH > (idealpH + 0.7)) {
        recommendations.push({
            type: 'rec-soil',
            text: '🧪 Soil Correction (High pH): Your soil is too alkaline. Add sulfur or peat moss to lower the pH.'
        });
    }

    // 5. Growth Improvement
    if (slowGrowth && (formData.leafColor === 'Yellow' || formData.leafColor === 'Pale')) {
         recommendations.push({
            type: 'rec-growth',
            text: '🌿 Growth Improvement: Model predicts Slow Growth and visual signs suggest possible Nitrogen deficiency. Recommend applying a balanced fertilizer.'
        });
    }

    if (recommendations.length === 0) {
        recommendations.push({
            type: 'rec-growth', 
            text: '👍 Everything looks good! Maintain current conditions for optimal health.'
        });
    }

    return recommendations;
}


// --- Growth and Flowering Insights ---
function generateInsights(formData, svmResults) {
    let insights = '';

    // Flowering Trend Insights
    const flowerCount = parseInt(formData.flowerCount);
    const isFloweringStage = formData.stage === 'Flowering' || formData.stage === 'Budding';

    insights += '<h4>🌸 Flowering Trend Insights</h4>';
    if (svmResults.HealthStatus === 'Healthy' && isFloweringStage && flowerCount > 5) {
        insights += '<p><strong>Excellent Potential:</strong> High flower/bud count indicates strong reproductive health. Maintain soil Phosphorus (P) levels.</p>';
    } else if (flowerCount === 0 && isFloweringStage) {
        insights += '<p><strong>Flowering Suppression:</strong> Zero flowers reported during a reproductive stage suggests severe stress or incorrect light exposure.</p>';
    } else {
        insights += '<p><strong>Observation:</strong> The current flower count is acceptable for the plant\'s health status and stage of growth.</p>';
    }

    // Growth Improvement Suggestions
    insights += '<h4>🌿 Growth Improvement Suggestions</h4>';
    const userHeight = parseInt(formData.plantHeight);
    const idealHeight = SPECIES_IDEALS[formData.species].height.ideal;
    
    if (userHeight < (idealHeight * 0.75)) {
        insights += `<p><strong>Stunted Growth Warning:</strong> Plant height (${userHeight} cm) is significantly below the species ideal (${idealHeight} cm). Check root development and nutrition.</p>`;
    } else if (formData.leafColor === 'Yellow' && formData.fertilizerUsed === 'No') {
        insights += '<p><strong>Fertilizer Need:</strong> Yellow leaves combined with no recent fertilizer application strongly suggest a **Macronutrient deficiency**.</p>';
    } else {
        insights += '<p><strong>Consistent Growth:</strong> Physical parameters appear balanced. Focus on providing stable conditions.</p>';
    }

    return insights;
}


// --- Chart.js Analytics (Fixed for Data Types and Initialization) ---

function renderAnalytics(formData, consistencyResult, pestScore) {
    // FIX: Destroy old charts to prevent errors when re-rendering
    ['factor-contribution-chart', 'ideal-vs-user-chart', 'pest-indicators-chart'].forEach(id => {
        const chart = Chart.getChart(id);
        if (chart) chart.destroy();
    });

    // 1. Factor Contribution (Pie Chart)
    const consistencyStatus = consistencyResult.isConsistent ? 1 : 4; // Inconsistent = high score (4)
    const leafColorScore = 4 - CATEGORICAL_MAPPING.leafColor[formData.leafColor]; // Lower is bad
    const wiltingScore = formData.wiltingSigns === 'Yes' ? 4 : 1;
    const environmentScore = (parseFloat(formData.moisture) < 40 || parseFloat(formData.ph) < 5.5 || parseFloat(formData.light) < 5) ? 3 : 1;
    
    const scaledPestScore = pestScore; // Max score is 4

    const data = [
        scaledPestScore, 
        leafColorScore + wiltingScore, 
        environmentScore, 
        consistencyStatus
    ];
    
    new Chart(document.getElementById('factor-contribution-chart'), {
        type: 'pie',
        data: {
            labels: ['Pest/Disease Indicator', 'Plant Health Inputs (Color/Wilting)', 'Environmental Stress (Env.)', 'Species Consistency Mismatch'],
            datasets: [{
                data: data,
                backgroundColor: ['#e74c3c', '#f39c12', '#3498db', '#9b59b6'],
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Diagnosis Factor Contribution Score (Higher = Worse)' },
                tooltip: { callbacks: { label: (context) => `${context.label}: ${data[context.dataIndex].toFixed(1)} (Score)` } }
            }
        }
    });


    // 2. Ideal vs User Inputs (Line Chart)
    const ideal = SPECIES_IDEALS[formData.species];
    // FIX: Ensure all are parsed as numbers for Chart.js
    const userInputs = [
        parseFloat(formData.moisture),
        parseFloat(formData.ph),
        parseFloat(formData.light),
        parseInt(formData.plantHeight) 
    ];
    const idealInputs = [
        ideal.moisture.ideal,
        ideal.pH.ideal,
        ideal.light.ideal,
        ideal.height.ideal 
    ];
    
    new Chart(document.getElementById('ideal-vs-user-chart'), {
        type: 'line',
        data: {
            labels: ['Soil Moisture (%)', 'Soil pH', 'Light (Hours/Day)', 'Plant Height (cm)'],
            datasets: [
                {
                    label: 'Your Input',
                    data: userInputs,
                    borderColor: '#FF69B4',
                    backgroundColor: 'rgba(255, 105, 180, 0.2)',
                    tension: 0.3
                },
                {
                    label: `${formData.species} Ideal`,
                    data: idealInputs,
                    borderColor: '#38761D',
                    backgroundColor: 'rgba(56, 118, 29, 0.2)',
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: false } },
            plugins: { title: { display: true, text: 'Ideal vs. User Environmental Inputs' } }
        }
    });

    // 3. Pest Indicators (Bar Chart)
    const pestLabels = ['White Powder', 'Holes in Leaves', 'Sticky Leaves', 'Visible Insects'];
    const pestData = [
        formData.whitePowder === 'Yes' ? 1 : 0,
        formData.holesInLeaves === 'Yes' ? 1 : 0,
        formData.stickyLeaves === 'Yes' ? 1 : 0,
        formData.insectVisibility !== 'None' ? 1 : 0
    ];

    new Chart(document.getElementById('pest-indicators-chart'), {
        type: 'bar',
        data: {
            labels: pestLabels,
            datasets: [{
                label: 'Presence (1 = Yes, 0 = No)',
                data: pestData,
                backgroundColor: pestData.map(val => val === 1 ? '#CC0000' : '#2ecc71'),
                borderColor: '#333',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: true, max: 1, ticks: { stepSize: 1 } } },
            plugins: { title: { display: true, text: 'Individual Pest and Disease Indicators' } }
        }
    });
}

// --- UI Management and Event Handlers ---

function showSection(sectionId) {
    SECTIONS.forEach(id => {
        const sectionEl = document.getElementById(id);
        if(sectionEl) sectionEl.classList.add('hidden');
    });
    const sectionEl = document.getElementById(sectionId);
    if(sectionEl) sectionEl.classList.remove('hidden');

    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('data-section') === sectionId) {
            link.classList.add('active');
        }
    });
    
    // Smooth scroll only if the element exists
    if(sectionEl) sectionEl.scrollIntoView({ behavior: 'smooth' });
}

document.addEventListener('DOMContentLoaded', () => {
    // Train the models as soon as the page loads
    trainAllModels();
    showSection('home-section');

    const form = document.getElementById('diagnosis-form');
    const startDiagnosisBtn = document.getElementById('start-diagnosis-btn');
    const showAnalyticsBtn = document.getElementById('show-analytics-btn');

    startDiagnosisBtn?.addEventListener('click', () => showSection('diagnosis-form-section'));
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            showSection(e.target.getAttribute('data-section'));
        });
    });
    showAnalyticsBtn?.addEventListener('click', () => showSection('analytics-section'));


    // Form Submission Handler 
    form?.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // 1. Run strict validation
        if (!validateAllInputs(form)) {
            console.error("Form validation failed. Please check inputs.");
            return;
        }

        // --- FIX FOR LINE 191 ERROR ---
        const resultsContentEl = document.getElementById('results-content');
        const loadingMessageEl = document.getElementById('loading-message');

        showSection('results-section');
        
        // This is the section that was crashing. It MUST work now that the HTML IDs are verified.
        if (resultsContentEl && loadingMessageEl) {
            resultsContentEl.classList.add('hidden');
            loadingMessageEl.classList.remove('hidden');
        }

        // Collect form data 
        const formData = {
            species: document.getElementById('species').value,
            stage: document.getElementById('stage').value,
            moisture: document.getElementById('moisture').value,
            ph: document.getElementById('ph').value,
            light: document.getElementById('light').value,
            fertilizerUsed: document.querySelector('input[name="fertilizerUsed"]:checked')?.value || 'No',
            leafColor: document.getElementById('leafColor').value,
            wiltingSigns: document.querySelector('input[name="wiltingSigns"]:checked')?.value || 'No',
            flowerCount: document.getElementById('flowerCount').value,
            plantHeight: document.getElementById('plantHeight').value,
            whitePowder: document.querySelector('input[name="whitePowder"]:checked')?.value || 'No',
            holesInLeaves: document.querySelector('input[name="holesInLeaves"]:checked')?.value || 'No',
            stickyLeaves: document.querySelector('input[name="stickyLeaves"]:checked')?.value || 'No',
            insectVisibility: document.getElementById('insectVisibility').value,
        };

        // --- 2. ML Prediction (SVM) ---
        const { featureVector, pestScore } = preprocessUserInput(formData);

        const healthStatus = SVM_MODELS['HealthStatus'].predict(featureVector);
        const growthType = SVM_MODELS['GrowthType'].predict(featureVector);
        const riskLevel = SVM_MODELS['RiskLevel'].predict(featureVector);

        const svmResults = { HealthStatus: healthStatus, GrowthType: growthType, RiskLevel: riskLevel };

        // --- 3. Species Consistency Check & Decision Engine ---
        const consistencyResult = runSpeciesConsistency(formData);
        const recommendations = runDecisionEngine(svmResults, formData);
        const insightsHTML = generateInsights(formData, svmResults);

        // --- 4. Render Results ---
        if (loadingMessageEl && resultsContentEl) {
            loadingMessageEl.classList.add('hidden');
            resultsContentEl.classList.remove('hidden');
        }

        // Main Predictions
        document.getElementById('result-health').textContent = svmResults.HealthStatus;
        document.getElementById('result-health').className = `health-status ${svmResults.HealthStatus.replace(' ', '')}`;

        document.getElementById('result-growth').textContent = svmResults.GrowthType;
        document.getElementById('result-growth').className = `growth-type ${svmResults.GrowthType.replace(' ', '')}`;

        document.getElementById('result-risk').textContent = svmResults.RiskLevel;
        document.getElementById('result-risk').className = `risk-level ${svmResults.RiskLevel.replace(' ', '')}`;

        // Consistency Check
        const consistencyCard = document.getElementById('consistency-check-card');
        const consistencyMsgEl = document.getElementById('result-consistency-message');
        const consistencyDetailsEl = document.getElementById('result-consistency-details');
        
        consistencyMsgEl.textContent = consistencyResult.isConsistent ? 'Consistent with Species Requirements' : 'Warning: Your entered conditions mismatch typical species needs.';
        consistencyCard.className = `result-card consistency-card ${consistencyResult.isConsistent ? 'Consistent' : 'Warning'}`;

        consistencyDetailsEl.innerHTML = consistencyResult.details.map(d => 
            `<li>${d.status}: ${d.label} (User: ${d.user}, Ideal: ${d.ideal})</li>`
        ).join('');

        // Recommendations
        document.getElementById('recommendations-container').innerHTML = recommendations.map(rec => 
            `<div class="recommendation-card ${rec.type}">${rec.text}</div>`
        ).join('');

        // Insights
        document.getElementById('growth-insights-container').innerHTML = insightsHTML;

        // --- 5. Render Analytics (Charts) ---
        renderAnalytics(formData, consistencyResult, pestScore);

        // Scroll to results
        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
    });

    // Attach real-time validation listeners
    document.querySelectorAll('#diagnosis-form input, #diagnosis-form select').forEach(input => {
        input.addEventListener('input', () => {
            if (input.type === 'number') {
                validateInput(input, input.min, input.max, 'number');
            } else if (input.tagName === 'SELECT') {
                validateInput(input, null, null, 'select');
            }
        });
    });
});