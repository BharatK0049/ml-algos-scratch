const { PythonShell } = require('python-shell');

// Dummy Data
const payload = {
    x: [1, 2, 3, 4, 5],
    y: [5, 7, 9, 11, 13], // Perfect 2x + 3
    lr: 0.01,
    epochs: 500
};

// 2. Configure the Options
let options = {
    mode: 'text',
    pythonOptions: ['-u'], // get print results in real-time
    args: [JSON.stringify(payload)] // Pass the data as a generic string
};

console.log("🚀 Launching Python script...");

// 3. Run Python
PythonShell.run('linear_regression.py', options).then(messages => {
    // messages is an array of strings (stdout)
    // The last message should be our JSON object
    
    try {
        const result = JSON.parse(messages[0]);
        
        if (result.status === 'success') {
            console.log("✅ Success!");
            console.log("Final Weight (Slope):", result.final_weights);
            console.log("Final Bias (Intercept):", result.final_bias);
            console.log("History Length:", result.history.length);
            console.log("Sample History (Epoch 0):", result.history[0]);
        } else {
            console.error("❌ Python Error:", result.message);
        }
    } catch (e) {
        console.error("❌ Failed to parse Python output:", e);
        console.log("Raw Output:", messages);
    }
});