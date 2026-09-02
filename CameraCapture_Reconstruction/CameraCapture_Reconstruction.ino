#include <TinyMLShield.h>

unsigned short pixels[176 * 144];  // QCIF RGB565

void setup()
{
  Serial.begin(921600);

  while (!Serial)
    ;

  Serial.println("Starting Camera...");

  if (!Camera.begin(QCIF, RGB565, 1, OV7675))
  {
    Serial.println("Camera init failed!");

    while (1)
      ;
  }

  Serial.println("Camera OK");

  delay(3000);

  Camera.readFrame(pixels);

  int numPixels = Camera.width() * Camera.height();

  Serial.print("Pixels = ");
  Serial.println(numPixels);

  Serial.println("HEXADECIMAL_BYTES = [");

  for (int i = 0; i < numPixels; i++)
  {
    Serial.print("0x");

    if (pixels[i] < 0x1000)
      Serial.print("0");

    if (pixels[i] < 0x0100)
      Serial.print("0");

    if (pixels[i] < 0x0010)
      Serial.print("0");

    Serial.print(pixels[i], HEX);

    if (i < numPixels - 1)
    {
      Serial.print(", ");
    }

    if ((i + 1) % 16 == 0)
    {
      Serial.println();
    }
  }

  Serial.println();
  Serial.println("]");

  Serial.println("Finished.");
}

void loop()
{
  while (1)
    ;  // stop after one frame
}