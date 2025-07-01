/**
 * 🎨 GUESS MY DRAWING - JavaScript
 * Logique frontend pour canvas et prédictions IA
 */

// Configuration
const CONFIG = {
    // Auto-détection de l'API (Vercel ou local)
    API_BASE_URL: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
        ? 'http://localhost:5000' 
        : '', // URL relative pour Vercel
    PREDICTION_DELAY: 2000, // 2 secondes
    CANVAS_SIZE: 400,
    IMAGE_SIZE: 28, // Taille pour le modèle
    MIN_STROKES: 5, // Minimum de traits avant prédiction
    BRUSH_COLOR: '#000000',
    BACKGROUND_COLOR: '#ffffff'
};

// Variables globales
let canvas = null;
let ctx = null;
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let strokeCount = 0;
let predictionTimer = null;
let drawingCount = 0;
let isConnected = false;

// Classes avec emojis et couleurs (sans dog pour éviter confusion)
const CLASSES_INFO = {
    'cat': { emoji: '🐱', color: '#FF6B6B' },
    'house': { emoji: '🏠', color: '#45B7D1' },
    'car': { emoji: '🚗', color: '#96CEB4' },
    'tree': { emoji: '🌳', color: '#FFEAA7' }
};

/**
 * 🚀 INITIALISATION
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎨 Guess My Drawing - Démarrage');
    
    initializeCanvas();
    initializeEventListeners();
    checkAPIConnection();
    
    console.log('✅ Interface initialisée');
});

/**
 * 🖼️ CANVAS - Initialisation
 */
function initializeCanvas() {
    canvas = document.getElementById('drawingCanvas');
    ctx = canvas.getContext('2d');
    
    // Configuration canvas
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = CONFIG.BRUSH_COLOR;
    ctx.lineWidth = 8;
    
    // Canvas haute résolution
    const rect = canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    
    canvas.width = CONFIG.CANVAS_SIZE * scale;
    canvas.height = CONFIG.CANVAS_SIZE * scale;
    
    ctx.scale(scale, scale);
    canvas.style.width = CONFIG.CANVAS_SIZE + 'px';
    canvas.style.height = CONFIG.CANVAS_SIZE + 'px';
    
    clearCanvas();
    showInstructions();
    
    console.log('🖼️ Canvas initialisé');
}

/**
 * 🎮 EVENT LISTENERS
 */
function initializeEventListeners() {
    // Canvas Events - Mouse
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Canvas Events - Touch
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    // Buttons
    document.getElementById('clearBtn').addEventListener('click', clearCanvas);
    document.getElementById('newBtn').addEventListener('click', newDrawing);
    
    // Brush size
    const brushSize = document.getElementById('brushSize');
    brushSize.addEventListener('input', function() {
        ctx.lineWidth = parseInt(this.value);
        document.getElementById('brushSizeValue').textContent = this.value + 'px';
    });
    
    console.log('🎮 Event listeners configurés');
}

/**
 * 🖊️ DRAWING - Start
 */
function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = getMousePos(e);
    
    hideInstructions();
    
    // Clear prediction timer
    if (predictionTimer) {
        clearTimeout(predictionTimer);
    }
}

/**
 * 🖊️ DRAWING - Draw
 */
function draw(e) {
    if (!isDrawing) return;
    
    const [currentX, currentY] = getMousePos(e);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();
    
    [lastX, lastY] = [currentX, currentY];
    strokeCount++;
    
    // Schedule prediction
    schedulePrediction();
}

/**
 * 🖊️ DRAWING - Stop
 */
function stopDrawing() {
    if (!isDrawing) return;
    isDrawing = false;
    
    // Final prediction
    schedulePrediction();
}

/**
 * 📱 TOUCH HANDLING
 */
function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                     e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

/**
 * 📍 MOUSE POSITION
 */
function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return [
        (e.clientX - rect.left) * scaleX / (window.devicePixelRatio || 1),
        (e.clientY - rect.top) * scaleY / (window.devicePixelRatio || 1)
    ];
}

/**
 * 🗑️ CLEAR CANVAS
 */
function clearCanvas() {
    ctx.fillStyle = CONFIG.BACKGROUND_COLOR;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    strokeCount = 0;
    
    // Clear predictions
    const container = document.getElementById('predictionsContainer');
    container.innerHTML = '<div class="predictions-placeholder">Commence à dessiner pour voir les prédictions !</div>';
    
    // Reset stats
    document.getElementById('maxConfidence').textContent = '-';
    document.getElementById('processingTime').textContent = '-';
    
    showInstructions();
    
    console.log('🗑️ Canvas effacé');
}

/**
 * ✨ NEW DRAWING
 */
function newDrawing() {
    clearCanvas();
    drawingCount++;
    document.getElementById('drawingCount').textContent = drawingCount;
    
    console.log(`✨ Nouveau dessin #${drawingCount}`);
}

/**
 * 💡 INSTRUCTIONS
 */
function showInstructions() {
    const instructions = document.getElementById('instructions');
    instructions.classList.add('show');
}

function hideInstructions() {
    const instructions = document.getElementById('instructions');
    instructions.classList.remove('show');
}

/**
 * ⏰ SCHEDULE PREDICTION
 */
function schedulePrediction() {
    if (predictionTimer) {
        clearTimeout(predictionTimer);
    }
    
    if (strokeCount >= CONFIG.MIN_STROKES) {
        predictionTimer = setTimeout(() => {
            requestPrediction();
        }, CONFIG.PREDICTION_DELAY);
    }
}

/**
 * 🤖 REQUEST PREDICTION
 */
async function requestPrediction() {
    if (!isConnected) {
        console.warn('⚠️ API non connectée');
        return;
    }
    
    try {
        setStatus('processing', 'Analyse en cours...');
        showLoading(true);
        
        // Convertir canvas en image 28x28
        const imageData = canvasToImageData();
        
        const startTime = Date.now();
        
        // Requête API
        const response = await fetch(`${CONFIG.API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData,
                timestamp: new Date().toISOString()
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const result = await response.json();
        const processingTime = Date.now() - startTime;
        
        displayPredictions(result.predictions);
        updateStats(result, processingTime);
        
        setStatus('online', 'IA connectée');
        
        console.log('🤖 Prédiction reçue:', result);
        
    } catch (error) {
        console.error('❌ Erreur prédiction:', error);
        setStatus('offline', 'Erreur IA');
        
        // Show error in predictions
        const container = document.getElementById('predictionsContainer');
        container.innerHTML = `
            <div class="error-message">
                ❌ Erreur de prédiction<br>
                <small>${error.message}</small>
            </div>
        `;
        
    } finally {
        showLoading(false);
    }
}

/**
 * 🖼️ CANVAS TO IMAGE DATA
 */
function canvasToImageData() {
    // Créer un canvas temporaire 28x28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = CONFIG.IMAGE_SIZE;
    tempCanvas.height = CONFIG.IMAGE_SIZE;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Fond blanc
    tempCtx.fillStyle = '#ffffff';
    tempCtx.fillRect(0, 0, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE);
    
    // Redimensionner le dessin
    tempCtx.drawImage(canvas, 0, 0, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE);
    
    // Convertir en ImageData
    const imageData = tempCtx.getImageData(0, 0, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE);
    
    // Convertir en array pour l'API (format Quick Draw)
    const pixels = [];
    for (let i = 0; i < imageData.data.length; i += 4) {
        // Prendre seulement le canal rouge (image en niveaux de gris)
        // Inverser : Quick Draw format (0=blanc, 255=noir)
        const gray = 255 - imageData.data[i];
        pixels.push(gray);
    }
    
    return pixels;
}

/**
 * 📊 DISPLAY PREDICTIONS
 */
function displayPredictions(predictions) {
    const container = document.getElementById('predictionsContainer');
    
    if (!predictions || predictions.length === 0) {
        container.innerHTML = '<div class="predictions-placeholder">Aucune prédiction disponible</div>';
        return;
    }
    
    // Trier par confiance
    const sortedPredictions = [...predictions].sort((a, b) => b.confidence - a.confidence);
    
    let html = '';
    sortedPredictions.forEach((pred, index) => {
        const classInfo = CLASSES_INFO[pred.class] || { emoji: '❓', color: '#999' };
        const percentage = (pred.confidence * 100).toFixed(1);
        
        html += `
            <div class="prediction-item" style="animation-delay: ${index * 0.1}s">
                <div class="prediction-emoji">${classInfo.emoji}</div>
                <div class="prediction-info">
                    <div class="prediction-label">${pred.class.charAt(0).toUpperCase() + pred.class.slice(1)}</div>
                    <div class="prediction-bar">
                        <div class="prediction-fill" 
                             style="width: ${percentage}%; background-color: ${classInfo.color};">
                        </div>
                    </div>
                </div>
                <div class="prediction-confidence">${percentage}%</div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

/**
 * 📈 UPDATE STATS
 */
function updateStats(result, processingTime) {
    // Processing time
    document.getElementById('processingTime').textContent = `${processingTime}ms`;
    
    // Max confidence
    if (result.predictions && result.predictions.length > 0) {
        const maxConf = Math.max(...result.predictions.map(p => p.confidence));
        document.getElementById('maxConfidence').textContent = `${(maxConf * 100).toFixed(1)}%`;
    }
}

/**
 * 🔌 CHECK API CONNECTION
 */
async function checkAPIConnection() {
    try {
        setStatus('processing', 'Connexion...');
        
        const response = await fetch(`${CONFIG.API_BASE_URL}/health`);
        
        if (response.ok) {
            const data = await response.json();
            isConnected = true;
            setStatus('online', 'IA connectée');
            
            console.log('✅ API connectée:', data);
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
        
    } catch (error) {
        console.warn('⚠️ API non disponible:', error);
        isConnected = false;
        setStatus('offline', 'IA déconnectée');
    }
}

/**
 * 🚨 SET STATUS
 */
function setStatus(type, message) {
    const indicator = document.getElementById('statusIndicator');
    const text = document.getElementById('statusText');
    
    indicator.className = `status-indicator ${type}`;
    text.textContent = message;
}

/**
 * ⏳ LOADING OVERLAY
 */
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (show) {
        overlay.classList.add('show');
    } else {
        overlay.classList.remove('show');
    }
}

/**
 * 🔄 AUTO RECONNECT
 */
setInterval(() => {
    if (!isConnected) {
        checkAPIConnection();
    }
}, 10000); // Retry every 10 seconds

// Debug
window.DEBUG = {
    canvas,
    ctx,
    CONFIG,
    requestPrediction,
    clearCanvas,
    checkAPIConnection
}; 