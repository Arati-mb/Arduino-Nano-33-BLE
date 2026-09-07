#include <Arduino.h>
#include "model.h"

#include "TensorFlowLite.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

const int INPUT_WIDTH = 96;
const int INPUT_HEIGHT = 96;
const int INPUT_CHANNELS = 3;

const int NUM_CLASSES = 11;

const char* labels[NUM_CLASSES] = {
  "0-20",
  "0-50",
  "101-150",
  "151-200",
  "201-250",
  "21-40",
  "251-300",
  "41-60",
  "51-100",
  "61-80",
  "81-100"
};

tflite::ErrorReporter* error_reporter = nullptr;

const tflite::Model* model_data = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;

constexpr int TENSOR_ARENA_SIZE = 100 * 1024;

uint8_t tensor_arena[TENSOR_ARENA_SIZE];

void setup()
{
  Serial.begin(115200);

  while (!Serial)
  {
    delay(10);
  }

  Serial.println();
  Serial.println("================================");
  Serial.println(" Gauge TinyML Deployment");
  Serial.println(" Arduino Nano 33 BLE Sense");
  Serial.println("================================");

  static tflite::MicroErrorReporter micro_error_reporter;

  error_reporter = &micro_error_reporter;

  Serial.println("Loading model...");

  model_data = tflite::GetModel(model);

  if (model_data == nullptr)
  {
    Serial.println("ERROR: Model is NULL");
    return;
  }

  if (model_data->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.print("ERROR: Model schema version: ");
    Serial.println(model_data->version());

    Serial.print("Expected schema version: ");
    Serial.println(TFLITE_SCHEMA_VERSION);

    return;
  }

  Serial.println("Model loaded successfully.");

static tflite::MicroMutableOpResolver<6> resolver;

resolver.AddConv2D();
resolver.AddMaxPool2D();
resolver.AddFullyConnected();
resolver.AddSoftmax();
resolver.AddReshape();
resolver.AddMean();

  static tflite::MicroInterpreter static_interpreter(
    model_data,
    resolver,
    tensor_arena,
    TENSOR_ARENA_SIZE,
    error_reporter
  );

  interpreter = &static_interpreter;

  Serial.println("Allocating tensors...");

  TfLiteStatus status = interpreter->AllocateTensors();

  if (status != kTfLiteOk)
  {
    Serial.println("ERROR: AllocateTensors failed.");
    Serial.println("Increase TENSOR_ARENA_SIZE.");
    return;
  }

  Serial.println("Tensors allocated successfully.");

  input = interpreter->input(0);

  if (input == nullptr)
  {
    Serial.println("ERROR: Input tensor is NULL.");
    return;
  }

  Serial.println();
  Serial.println("Input tensor:");

  Serial.print("Type: ");
  Serial.println(input->type);

  Serial.print("Bytes: ");
  Serial.println(input->bytes);

  Serial.print("Dimensions: ");

  for (int i = 0; i < input->dims->size; i++)
  {
    Serial.print(input->dims->data[i]);

    if (i < input->dims->size - 1)
    {
      Serial.print(" x ");
    }
  }

  Serial.println();

  Serial.println();
  Serial.println("================================");
  Serial.println(" Model ready!");
  Serial.println("================================");

  runInference();
}

void loop()
{
  delay(5000);

  runInference();
}

void runInference()
{
  if (input == nullptr)
  {
    Serial.println("ERROR: Input tensor unavailable.");
    return;
  }

  Serial.println();
  Serial.println("Running test inference...");

  /*
     TEST IMAGE

     For now we fill the input with zero values.
     This is only to check that the model can
     successfully run on the Arduino.

     Later this will be replaced with the
     actual camera image.
  */

  if (input->type == kTfLiteInt8)
  {
    for (int i = 0; i < input->bytes; i++)
    {
      input->data.int8[i] = 0;
    }
  }
  else
  {
    Serial.println("ERROR: Model input is not INT8.");
    return;
  }

  unsigned long start_time = millis();

  TfLiteStatus status = interpreter->Invoke();

  unsigned long inference_time = millis() - start_time;

  if (status != kTfLiteOk)
  {
    Serial.println("ERROR: Inference failed.");
    return;
  }

  Serial.print("Inference time: ");
  Serial.print(inference_time);
  Serial.println(" ms");

  TfLiteTensor* output = interpreter->output(0);

  if (output == nullptr)
  {
    Serial.println("ERROR: Output tensor is NULL.");
    return;
  }

  int predicted_class = 0;

  int8_t highest_value = -128;

  Serial.println();
  Serial.println("Predictions:");

  for (int i = 0; i < NUM_CLASSES; i++)
  {
    int8_t value = output->data.int8[i];

    Serial.print(labels[i]);
    Serial.print(": ");

    Serial.println(value);

    if (value > highest_value)
    {
      highest_value = value;
      predicted_class = i;
    }
  }

  Serial.println();

  Serial.println("============================");

  Serial.print("Predicted pressure: ");
  Serial.println(labels[predicted_class]);

  Serial.print("Quantized score: ");
  Serial.println(highest_value);

  Serial.println("============================");
}