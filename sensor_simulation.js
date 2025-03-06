// Constants based on dataset
const MIN_TEMP = 45;
const MAX_TEMP = 80;
const STUCK_REPEAT_COUNT = 8;  // Number of stuck-at fault readings
const NORMAL_COUNT = 10;       // Number of normal readings before faults start

// Fault tracking variables
let readingCount = context.get("readingCount") || 0;
let stuckTemperature = context.get("stuckTemperature") || null;
let driftTemperature = context.get("driftTemperature") || null;
let intermittentTemperature = context.get("intermittentTemperature") || null;
let driftCounter = context.get("driftCounter") || 0;

// Function to generate normal sensor data
function generateNormalSensorData() {
    let temperature = (Math.random() * (MAX_TEMP - MIN_TEMP) + MIN_TEMP).toFixed(2);
    return {
        sensor_id: "DHT22-1",
        temperature: parseFloat(temperature),
        timestamp: new Date().toISOString(),
        fault_type: "normal"
    };
}

// Function to generate faulty sensor data
function generateFaultySensorData() {
    let temperature;
    let faultType = "normal";

    if (readingCount < NORMAL_COUNT) {
        // First 10 readings: Normal
        let data = generateNormalSensorData();
        temperature = data.temperature;
        faultType = data.fault_type;
    } 
    else if (readingCount < NORMAL_COUNT + STUCK_REPEAT_COUNT) {
        // Stuck-at Fault (next 8 readings)
        if (stuckTemperature === null) {
            stuckTemperature = (Math.random() * (MAX_TEMP - MIN_TEMP) + MIN_TEMP).toFixed(2);
            context.set("stuckTemperature", stuckTemperature);
        }
        temperature = stuckTemperature;
        faultType = "stuck_at";
    } 
    else if (driftCounter < 10) {
        // Drift Fault (gradual increase)
        if (driftTemperature === null) {
            driftTemperature = (Math.random() * (MAX_TEMP - MIN_TEMP) + MIN_TEMP);
            context.set("driftTemperature", driftTemperature);
        }
        driftTemperature += (Math.random() * 2); // Small gradual increase
        temperature = driftTemperature.toFixed(2);
        driftCounter++;
        faultType = "drift";
        context.set("driftCounter", driftCounter);
    } 
    else {
        // Introduce random faults while keeping sequence
        let faultTypeIndex = Math.floor(Math.random() * 4);

        switch (faultTypeIndex) {
            case 0: // Noise Fault (random minor changes)
                if (Math.random() < 0.2) { // 20% chance for noise
                    temperature = (parseFloat(generateNormalSensorData().temperature) + (Math.random() * 3 - 1.5)).toFixed(2);
                    faultType = "noise";
                } else {
                    let data = generateNormalSensorData();
                    temperature = data.temperature;
                    faultType = data.fault_type;
                }
                break;

            case 1: // Out-of-Range Fault (Extreme Values)
                if (Math.random() < 0.1) { // 10% chance for extreme values
                    temperature = (Math.random() * 150 + 50).toFixed(2);
                    faultType = "out_of_range";
                } else {
                    let data = generateNormalSensorData();
                    temperature = data.temperature;
                    faultType = data.fault_type;
                }
                break;

            case 2: // Intermittent Fault (Repeated Readings)
                if (intermittentTemperature === null || Math.random() < 0.2) { // 20% chance to reset
                    intermittentTemperature = (Math.random() * (MAX_TEMP - MIN_TEMP) + MIN_TEMP).toFixed(2);
                    context.set("intermittentTemperature", intermittentTemperature);
                }
                temperature = intermittentTemperature;
                faultType = "intermittent";
                break;

            case 3: // Calibration Fault (Offset Error)
                if (Math.random() < 0.15) { // 15% chance to add offset
                    temperature = (parseFloat(generateNormalSensorData().temperature) + 10).toFixed(2);
                    faultType = "calibration";
                } else {
                    let data = generateNormalSensorData();
                    temperature = data.temperature;
                    faultType = data.fault_type;
                }
                break;

            default:
                let data = generateNormalSensorData();
                temperature = data.temperature;
                faultType = data.fault_type;
        }
    }

    readingCount++; // Increment reading counter
    context.set("readingCount", readingCount);

    return {
        sensor_id: "DHT22-1",
        temperature: parseFloat(temperature),
        timestamp: new Date().toISOString(),
        fault_type: faultType
    };
}

// Function to send sensor data every 2 seconds
function sendSensorData() {
    let sensorData = generateFaultySensorData();
    
    node.send([
        { payload: sensorData }  // Send structured fault sequence
    ]);
}

// Execute every 2 seconds (controlled by Inject node)
sendSensorData();

return null;
