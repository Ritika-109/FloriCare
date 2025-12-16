// dataset.js

// --- 1. SPECIES IDEAL PROFILES for Consistency Check ---
const SPECIES_IDEALS = {
    'Rose': {
        moisture: { ideal: 60, min: 45, max: 75, unit: '%' },
        pH: { ideal: 6.5, min: 6.0, max: 7.0, unit: 'pH' },
        light: { ideal: 8, min: 6, max: 10, unit: 'hours/day' },
        height: { ideal: 60, min: 40, max: 100, unit: 'cm' },
    },
    'Marigold': {
        moisture: { ideal: 50, min: 35, max: 65, unit: '%' },
        pH: { ideal: 6.0, min: 5.5, max: 6.5, unit: 'pH' },
        light: { ideal: 7, min: 5, max: 9, unit: 'hours/day' },
        height: { ideal: 40, min: 25, max: 60, unit: 'cm' },
    },
    'Jasmine': {
        moisture: { ideal: 70, min: 55, max: 85, unit: '%' },
        pH: { ideal: 6.0, min: 5.0, max: 7.0, unit: 'pH' },
        light: { ideal: 6, min: 4, max: 8, unit: 'hours/day' },
        height: { ideal: 120, min: 80, max: 200, unit: 'cm' },
    },
    'Sunflower': {
        moisture: { ideal: 55, min: 40, max: 70, unit: '%' },
        pH: { ideal: 6.8, min: 6.0, max: 7.5, unit: 'pH' },
        light: { ideal: 10, min: 8, max: 12, unit: 'hours/day' },
        height: { ideal: 150, min: 100, max: 300, unit: 'cm' },
    },
    'Hibiscus': {
        moisture: { ideal: 65, min: 50, max: 80, unit: '%' },
        pH: { ideal: 6.2, min: 5.5, max: 7.0, unit: 'pH' },
        light: { ideal: 9, min: 7, max: 11, unit: 'hours/day' },
        height: { ideal: 80, min: 50, max: 150, unit: 'cm' },
    },
    'Tulip': {
        moisture: { ideal: 45, min: 30, max: 60, unit: '%' },
        pH: { ideal: 6.5, min: 6.0, max: 7.5, unit: 'pH' },
        light: { ideal: 5, min: 3, max: 7, unit: 'hours/day' },
        height: { ideal: 30, min: 15, max: 45, unit: 'cm' },
    }
};

// --- 2. SYNTHETIC TRAINING DATASET (Reduced Feature Set) ---
// Note: This is a small sample. In a real application, this would be hundreds of records.
const FLOWER_DATASET = [
    // Features: species, stage, moisture, pH, light, fertilizer (1/0), leafColor, wilting (1/0), flowerCount, height, pestScore (0-4), HealthStatus, GrowthType, RiskLevel
    { species: 'Rose', stage: 'Flowering', moisture: 60, pH: 6.5, light: 8, fertilizer: 1, leafColor: 'Normal', wilting: 0, flowerCount: 15, height: 65, pestScore: 0, HealthStatus: 'Healthy', GrowthType: 'Normal', RiskLevel: 'Low' },
    { species: 'Rose', stage: 'Vegetative', moisture: 75, pH: 5.5, light: 6, fertilizer: 0, leafColor: 'Yellow', wilting: 1, flowerCount: 0, height: 40, pestScore: 3, HealthStatus: 'Unhealthy', GrowthType: 'Slow', RiskLevel: 'High' },
    { species: 'Marigold', stage: 'Budding', moisture: 45, pH: 6.2, light: 7, fertilizer: 1, leafColor: 'Normal', wilting: 0, flowerCount: 5, height: 45, pestScore: 1, HealthStatus: 'Healthy', GrowthType: 'Fast', RiskLevel: 'Low' },
    { species: 'Marigold', stage: 'Seedling', moisture: 30, pH: 7.0, light: 5, fertilizer: 0, leafColor: 'Pale', wilting: 1, flowerCount: 0, height: 10, pestScore: 0, HealthStatus: 'Unhealthy', GrowthType: 'Slow', RiskLevel: 'Medium' },
    { species: 'Jasmine', stage: 'Flowering', moisture: 80, pH: 6.0, light: 5, fertilizer: 1, leafColor: 'Dark Green', wilting: 0, flowerCount: 10, height: 130, pestScore: 2, HealthStatus: 'Moderate', GrowthType: 'Normal', RiskLevel: 'Medium' },
    { species: 'Jasmine', stage: 'Vegetative', moisture: 65, pH: 7.5, light: 8, fertilizer: 0, leafColor: 'Yellow', wilting: 1, flowerCount: 3, height: 90, pestScore: 3, HealthStatus: 'Unhealthy', GrowthType: 'Slow', RiskLevel: 'High' },
    { species: 'Sunflower', stage: 'Budding', moisture: 55, pH: 6.8, light: 10, fertilizer: 1, leafColor: 'Normal', wilting: 0, flowerCount: 8, height: 160, pestScore: 0, HealthStatus: 'Healthy', GrowthType: 'Fast', RiskLevel: 'Low' },
    { species: 'Sunflower', stage: 'Seedling', moisture: 40, pH: 5.8, light: 12, fertilizer: 0, leafColor: 'Pale', wilting: 0, flowerCount: 0, height: 50, pestScore: 1, HealthStatus: 'Moderate', GrowthType: 'Normal', RiskLevel: 'Medium' },
    { species: 'Hibiscus', stage: 'Flowering', moisture: 65, pH: 6.2, light: 9, fertilizer: 1, leafColor: 'Normal', wilting: 0, flowerCount: 12, height: 85, pestScore: 0, HealthStatus: 'Healthy', GrowthType: 'Normal', RiskLevel: 'Low' },
    { species: 'Hibiscus', stage: 'Vegetative', moisture: 85, pH: 5.0, light: 7, fertilizer: 0, leafColor: 'Dark Green', wilting: 1, flowerCount: 0, height: 55, pestScore: 2, HealthStatus: 'Unhealthy', GrowthType: 'Slow', RiskLevel: 'High' },
    { species: 'Tulip', stage: 'Budding', moisture: 45, pH: 6.5, light: 5, fertilizer: 1, leafColor: 'Normal', wilting: 0, flowerCount: 3, height: 35, pestScore: 0, HealthStatus: 'Healthy', GrowthType: 'Normal', RiskLevel: 'Low' },
    { species: 'Tulip', stage: 'Seedling', moisture: 35, pH: 7.5, light: 3, fertilizer: 0, leafColor: 'Pale', wilting: 1, flowerCount: 0, height: 15, pestScore: 1, HealthStatus: 'Unhealthy', GrowthType: 'Slow', RiskLevel: 'Medium' },
    // More samples for robustness
    { species: 'Rose', stage: 'Flowering', moisture: 50, pH: 6.8, light: 7, fertilizer: 1, leafColor: 'Normal', wilting: 0, flowerCount: 10, height: 60, pestScore: 0, HealthStatus: 'Healthy', GrowthType: 'Normal', RiskLevel: 'Low' },
    { species: 'Sunflower', stage: 'Vegetative', moisture: 50, pH: 6.9, light: 11, fertilizer: 1, leafColor: 'Normal', wilting: 0, flowerCount: 0, height: 140, pestScore: 0, HealthStatus: 'Healthy', GrowthType: 'Fast', RiskLevel: 'Low' },
    { species: 'Marigold', stage: 'Flowering', moisture: 70, pH: 5.0, light: 3, fertilizer: 0, leafColor: 'Yellow', wilting: 1, flowerCount: 5, height: 30, pestScore: 4, HealthStatus: 'Unhealthy', GrowthType: 'Slow', RiskLevel: 'High' },
    { species: 'Hibiscus', stage: 'Budding', moisture: 80, pH: 5.5, light: 7, fertilizer: 0, leafColor: 'Dark Green', wilting: 0, flowerCount: 5, height: 70, pestScore: 2, HealthStatus: 'Moderate', GrowthType: 'Slow', RiskLevel: 'Medium' },
    { species: 'Tulip', stage: 'Flowering', moisture: 30, pH: 7.5, light: 2, fertilizer: 0, leafColor: 'Yellow', wilting: 1, flowerCount: 1, height: 20, pestScore: 3, HealthStatus: 'Unhealthy', GrowthType: 'Slow', RiskLevel: 'High' },
];