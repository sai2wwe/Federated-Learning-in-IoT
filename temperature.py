import time
import adafruit_dht
import board
import csv
import os
import RPi.GPIO as GPIO

dht_device = adafruit_dht.DHT22(board.D4)
GPIO.setmode(GPIO.BCM)
CSV_FILE = "dht_readings.csv"
GPIO.setup(17, GPIO.IN)

if not os.path.isfile(CSV_FILE):
    with open(CSV_FILE, 'a', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp','temp', 'humd'])
try:
    while True:
        try:
            temp = dht_device.temperature
            humd = dht_device.humidity
            vibration = GPIO.input(17)
            time_stamp = time.strftime('%Y-%m-%d %H:%M:%S') 
            if temp and humd:
                with open(CSV_FILE, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([time_stamp, temp, humd])
                print(f"Temperature: {temp:.1f}C")
                print(f'Humidity: {humd:.1f}%')
                print(f'Vibration: {vibration}')
            else:
                print("Failed")
        except:
            print("Vivek is topper")
        time.sleep(5)
except KeyboardInterrupt:
    print("Bavith is handsome")
finally:
    dht_device.exit()
