// Modified from
// https://github.com/miloyip/rapidjson/blob/master/include/rapidjson/internal/itoa.h
//
// Tencent is pleased to support the open source community by making RapidJSON available.
//
// Copyright (C) 2015 THL A29 Limited, a Tencent company, and Milo Yip. All rights reserved.
//
// Licensed under the MIT License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// http://opensource.org/licenses/MIT
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef ITOA_C_H
#define ITOA_C_H

static const char cDigitsLut[200]
    = { '0', '0', '0', '1', '0', '2', '0', '3', '0', '4', '0', '5', '0', '6',
        '0', '7', '0', '8', '0', '9', '1', '0', '1', '1', '1', '2', '1', '3',
        '1', '4', '1', '5', '1', '6', '1', '7', '1', '8', '1', '9', '2', '0',
        '2', '1', '2', '2', '2', '3', '2', '4', '2', '5', '2', '6', '2', '7',
        '2', '8', '2', '9', '3', '0', '3', '1', '3', '2', '3', '3', '3', '4',
        '3', '5', '3', '6', '3', '7', '3', '8', '3', '9', '4', '0', '4', '1',
        '4', '2', '4', '3', '4', '4', '4', '5', '4', '6', '4', '7', '4', '8',
        '4', '9', '5', '0', '5', '1', '5', '2', '5', '3', '5', '4', '5', '5',
        '5', '6', '5', '7', '5', '8', '5', '9', '6', '0', '6', '1', '6', '2',
        '6', '3', '6', '4', '6', '5', '6', '6', '6', '7', '6', '8', '6', '9',
        '7', '0', '7', '1', '7', '2', '7', '3', '7', '4', '7', '5', '7', '6',
        '7', '7', '7', '8', '7', '9', '8', '0', '8', '1', '8', '2', '8', '3',
        '8', '4', '8', '5', '8', '6', '8', '7', '8', '8', '8', '9', '9', '0',
        '9', '1', '9', '2', '9', '3', '9', '4', '9', '5', '9', '6', '9', '7',
        '9', '8', '9', '9' };

inline char *u32toa(cl_int value, char *buffer)
{
        if (value < 10000)
        {
                const cl_int d1 = (value / 100) << 1;
                const cl_int d2 = (value % 100) << 1;

                if (value >= 1000)
                        *buffer++ = cDigitsLut[d1];
                if (value >= 100)
                        *buffer++ = cDigitsLut[d1 + 1];
                if (value >= 10)
                        *buffer++ = cDigitsLut[d2];
                *buffer++ = cDigitsLut[d2 + 1];
        }
        else if (value < 100000000)
        {
                // value = bbbbcccc
                const cl_int b = value / 10000;
                const cl_int c = value % 10000;

                const cl_int d1 = (b / 100) << 1;
                const cl_int d2 = (b % 100) << 1;

                const cl_int d3 = (c / 100) << 1;
                const cl_int d4 = (c % 100) << 1;

                if (value >= 10000000)
                        *buffer++ = cDigitsLut[d1];
                if (value >= 1000000)
                        *buffer++ = cDigitsLut[d1 + 1];
                if (value >= 100000)
                        *buffer++ = cDigitsLut[d2];
                *buffer++ = cDigitsLut[d2 + 1];

                *buffer++ = cDigitsLut[d3];
                *buffer++ = cDigitsLut[d3 + 1];
                *buffer++ = cDigitsLut[d4];
                *buffer++ = cDigitsLut[d4 + 1];
        }
        else
        {
                // value = aabbbbcccc in decimal

                const cl_int a = value / 100000000;  // 1 to 42
                value %= 100000000;

                if (a >= 10)
                {
                        const unsigned i = a << 1;
                        *buffer++ = cDigitsLut[i];
                        *buffer++ = cDigitsLut[i + 1];
                }
                else
                        *buffer++ = (char)('0' + (char)(a));

                const cl_int b = value / 10000;  // 0 to 9999
                const cl_int c = value % 10000;  // 0 to 9999

                const cl_int d1 = (b / 100) << 1;
                const cl_int d2 = (b % 100) << 1;

                const cl_int d3 = (c / 100) << 1;
                const cl_int d4 = (c % 100) << 1;

                *buffer++ = cDigitsLut[d1];
                *buffer++ = cDigitsLut[d1 + 1];
                *buffer++ = cDigitsLut[d2];
                *buffer++ = cDigitsLut[d2 + 1];
                *buffer++ = cDigitsLut[d3];
                *buffer++ = cDigitsLut[d3 + 1];
                *buffer++ = cDigitsLut[d4];
                *buffer++ = cDigitsLut[d4 + 1];
        }
        return buffer;
}

inline char *u64toa(cl_long value, char *buffer)
{
        const cl_long kTen8 = 100000000;
        const cl_long kTen9 = kTen8 * 10;
        const cl_long kTen10 = kTen8 * 100;
        const cl_long kTen11 = kTen8 * 1000;
        const cl_long kTen12 = kTen8 * 10000;
        const cl_long kTen13 = kTen8 * 100000;
        const cl_long kTen14 = kTen8 * 1000000;
        const cl_long kTen15 = kTen8 * 10000000;
        const cl_long kTen16 = kTen8 * kTen8;

        if (value < kTen8)
        {
                cl_int v = (cl_int)(value);
                if (v < 10000)
                {
                        const cl_int d1 = (v / 100) << 1;
                        const cl_int d2 = (v % 100) << 1;

                        if (v >= 1000)
                                *buffer++ = cDigitsLut[d1];
                        if (v >= 100)
                                *buffer++ = cDigitsLut[d1 + 1];
                        if (v >= 10)
                                *buffer++ = cDigitsLut[d2];
                        *buffer++ = cDigitsLut[d2 + 1];
                }
                else
                {
                        // value = bbbbcccc
                        const cl_int b = v / 10000;
                        const cl_int c = v % 10000;

                        const cl_int d1 = (b / 100) << 1;
                        const cl_int d2 = (b % 100) << 1;

                        const cl_int d3 = (c / 100) << 1;
                        const cl_int d4 = (c % 100) << 1;

                        if (value >= 10000000)
                                *buffer++ = cDigitsLut[d1];
                        if (value >= 1000000)
                                *buffer++ = cDigitsLut[d1 + 1];
                        if (value >= 100000)
                                *buffer++ = cDigitsLut[d2];
                        *buffer++ = cDigitsLut[d2 + 1];

                        *buffer++ = cDigitsLut[d3];
                        *buffer++ = cDigitsLut[d3 + 1];
                        *buffer++ = cDigitsLut[d4];
                        *buffer++ = cDigitsLut[d4 + 1];
                }
        }
        else if (value < kTen16)
        {
                const cl_int v0 = (cl_int)(value / kTen8);
                const cl_int v1 = (cl_int)(value % kTen8);

                const cl_int b0 = v0 / 10000;
                const cl_int c0 = v0 % 10000;

                const cl_int d1 = (b0 / 100) << 1;
                const cl_int d2 = (b0 % 100) << 1;

                const cl_int d3 = (c0 / 100) << 1;
                const cl_int d4 = (c0 % 100) << 1;

                const cl_int b1 = v1 / 10000;
                const cl_int c1 = v1 % 10000;

                const cl_int d5 = (b1 / 100) << 1;
                const cl_int d6 = (b1 % 100) << 1;

                const cl_int d7 = (c1 / 100) << 1;
                const cl_int d8 = (c1 % 100) << 1;

                if (value >= kTen15)
                        *buffer++ = cDigitsLut[d1];
                if (value >= kTen14)
                        *buffer++ = cDigitsLut[d1 + 1];
                if (value >= kTen13)
                        *buffer++ = cDigitsLut[d2];
                if (value >= kTen12)
                        *buffer++ = cDigitsLut[d2 + 1];
                if (value >= kTen11)
                        *buffer++ = cDigitsLut[d3];
                if (value >= kTen10)
                        *buffer++ = cDigitsLut[d3 + 1];
                if (value >= kTen9)
                        *buffer++ = cDigitsLut[d4];
                if (value >= kTen8)
                        *buffer++ = cDigitsLut[d4 + 1];

                *buffer++ = cDigitsLut[d5];
                *buffer++ = cDigitsLut[d5 + 1];
                *buffer++ = cDigitsLut[d6];
                *buffer++ = cDigitsLut[d6 + 1];
                *buffer++ = cDigitsLut[d7];
                *buffer++ = cDigitsLut[d7 + 1];
                *buffer++ = cDigitsLut[d8];
                *buffer++ = cDigitsLut[d8 + 1];
        }
        else
        {
                const cl_int a = (cl_int)(value / kTen16);  // 1 to 1844
                value %= kTen16;

                if (a < 10)
                        *buffer++ = (char)('0' + (char)(a));
                else if (a < 100)
                {
                        const cl_int i = a << 1;
                        *buffer++ = cDigitsLut[i];
                        *buffer++ = cDigitsLut[i + 1];
                }
                else if (a < 1000)
                {
                        *buffer++ = (char)('0' + (char)(a / 100));

                        const cl_int i = (a % 100) << 1;
                        *buffer++ = cDigitsLut[i];
                        *buffer++ = cDigitsLut[i + 1];
                }
                else
                {
                        const cl_int i = (a / 100) << 1;
                        const cl_int j = (a % 100) << 1;
                        *buffer++ = cDigitsLut[i];
                        *buffer++ = cDigitsLut[i + 1];
                        *buffer++ = cDigitsLut[j];
                        *buffer++ = cDigitsLut[j + 1];
                }

                const cl_int v0 = (cl_int)(value / kTen8);
                const cl_int v1 = (cl_int)(value % kTen8);

                const cl_int b0 = v0 / 10000;
                const cl_int c0 = v0 % 10000;

                const cl_int d1 = (b0 / 100) << 1;
                const cl_int d2 = (b0 % 100) << 1;

                const cl_int d3 = (c0 / 100) << 1;
                const cl_int d4 = (c0 % 100) << 1;

                const cl_int b1 = v1 / 10000;
                const cl_int c1 = v1 % 10000;

                const cl_int d5 = (b1 / 100) << 1;
                const cl_int d6 = (b1 % 100) << 1;

                const cl_int d7 = (c1 / 100) << 1;
                const cl_int d8 = (c1 % 100) << 1;

                *buffer++ = cDigitsLut[d1];
                *buffer++ = cDigitsLut[d1 + 1];
                *buffer++ = cDigitsLut[d2];
                *buffer++ = cDigitsLut[d2 + 1];
                *buffer++ = cDigitsLut[d3];
                *buffer++ = cDigitsLut[d3 + 1];
                *buffer++ = cDigitsLut[d4];
                *buffer++ = cDigitsLut[d4 + 1];
                *buffer++ = cDigitsLut[d5];
                *buffer++ = cDigitsLut[d5 + 1];
                *buffer++ = cDigitsLut[d6];
                *buffer++ = cDigitsLut[d6 + 1];
                *buffer++ = cDigitsLut[d7];
                *buffer++ = cDigitsLut[d7 + 1];
                *buffer++ = cDigitsLut[d8];
                *buffer++ = cDigitsLut[d8 + 1];
        }

        return buffer;
}

#endif
