# Copyright (c) 2022 Robert Bosch GmbH and Microsoft Corporation
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""A sample skeleton vehicle app."""
import asyncio
import json
import logging
import signal

import torch
import torch.nn as nn
from vehicle import Vehicle, vehicle  # type: ignore
from velocitas_sdk.util.log import (  # type: ignore
    get_opentelemetry_log_factory,
    get_opentelemetry_log_format,
)
from velocitas_sdk.vdb.reply import DataPointReply
from velocitas_sdk.vehicle_app import VehicleApp

# Configure the VehicleApp logger with the necessary log config  and level.
logging.setLogRecordFactory(get_opentelemetry_log_factory())
logging.basicConfig(format=get_opentelemetry_log_format())
logging.getLogger().setLevel("DEBUG")
logger = logging.getLogger(__name__)

GET_SCORE_REQUEST_TOPIC = "sampleapp/getScore"
GET_SCORE_RESPONSE_TOPIC = "sampleapp/getScore/response"

DATABROKER_SPEED_SUBSCRIPTION_TOPIC = "sampleapp/currentSpeed"
DATABROKER_STEER_SUBSCRIPTION_TOPIC = "sampleapp/currentSteer"
DATABROKER_TRHOT_SUBSCRIPTION_TOPIC = "sampleapp/currentThrottle"
DATABROKER_BRAKE_SUBSCRIPTION_TOPIC = "sampleapp/currentBrake"
DATABROKER_LANE_SUBSCRIPTION_TOPIC = "sampleapp/currentLane"


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_sequence = lstm_out[:, -1, :]
        predictions = self.linear(last_sequence)
        return predictions


class SampleApp(VehicleApp):
    """
    Sample skeleton vehicle app.

    The skeleton subscribes to a getSpeed MQTT topic
    to listen for incoming requests to get
    the current vehicle speed and publishes it to
    a response topic.

    It also subcribes to the VehicleDataBroker
    directly for updates of the
    Vehicle.Speed signal and publishes this
    information via another specific MQTT topic
    """

    def __init__(self, vehicle_client: Vehicle):
        super().__init__()
        lstm_path = "/workspace/app/src/trained_lstm_model.pth"
        self.Vehicle = vehicle_client
        self.carla_speed = 0.0
        self.carla_steering = 0
        self.carla_brake = 0
        self.carla_throttle = 0.0
        self.carla_lane = 0
        self.tensor_array = torch.zeros([100, 5])
        self.count = 0
        self.count2 = 0
        self.score = torch.tensor(0.0)
        self.mean_score = 0.0

        self.input_size = 5
        self.hidden_size = 50
        self.output_size = 1
        self.num_layers = 2
        self.dropout_rate = 0.2
        self.sequence_length = 100

        self.model = LSTMModel(
            self.input_size,
            self.hidden_size,
            self.output_size,
            self.num_layers,
            self.dropout_rate,
        )
        self.model.load_state_dict(torch.load(lstm_path))
        self.model.eval()

    async def on_start(self):
        await self.Vehicle.Speed.subscribe(self.on_speed_change)
        await self.Vehicle.Chassis.SteeringWheel.Angle.subscribe(
            self.on_steering_change
        )
        await self.Vehicle.OBD.ThrottlePosition.subscribe(self.on_throttle_change)
        await self.Vehicle.Chassis.Brake.PedalPosition.subscribe(self.on_brake_change)
        await self.Vehicle.ADAS.LaneDepartureDetection.IsWarning.subscribe(
            self.on_lane_change
        )
        await self.Vehicle.Speed.subscribe(self.tensor_change)

    async def on_speed_change(self, data: DataPointReply):
        self.carla_speed = data.get(self.Vehicle.Speed).value

    async def on_steering_change(self, data: DataPointReply):
        self.carla_steering = data.get(self.Vehicle.Chassis.SteeringWheel.Angle).value

    async def on_throttle_change(self, data: DataPointReply):
        self.carla_throttle = data.get(self.Vehicle.OBD.ThrottlePosition).value

    async def on_brake_change(self, data: DataPointReply):
        self.carla_brake = data.get(self.Vehicle.Chassis.Brake.PedalPosition).value

    async def on_lane_change(self, data: DataPointReply):
        self.carla_lane = data.get(
            self.Vehicle.ADAS.LaneDepartureDetection.IsWarning
        ).value

    async def tensor_change(self, data: DataPointReply):
        self.carla_speed = data.get(self.Vehicle.Speed).value

        more_elements = torch.tensor(
            [
                self.carla_speed,
                self.carla_steering,
                self.carla_brake,
                self.carla_throttle,
                self.carla_lane,
            ]
        )

        self.tensor_array[self.count] = more_elements
        self.count += 1
        if self.count == 100:
            self.count2 += 1

            input_tensor = self.tensor_array.unsqueeze(0)

            with torch.no_grad():
                self.score = self.model(input_tensor)
            self.count = 0
            self.tensor_array = torch.zeros([100, 5])
            score = self.score.item()
            self.mean_score += score
            mean_score = self.mean_score / self.count2
            msg = f"current score: {round(score, 2)} mean score:{round(mean_score, 2)}"
            response_data = {"message": msg}

            await self.publish_event(
                GET_SCORE_RESPONSE_TOPIC,
                json.dumps(response_data),
            )


async def main():
    """Main function"""
    logger.info("Starting SampleApp...")
    vehicle_app = SampleApp(vehicle)

    await vehicle_app.run()


LOOP = asyncio.get_event_loop()
LOOP.add_signal_handler(signal.SIGTERM, LOOP.stop)
LOOP.run_until_complete(main())
LOOP.close()
