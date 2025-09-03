/*
负责读取和打印输入参数。Input.h 中声明了 ReadInput 和 PrintInput 函数，Input.cpp 实现了这两个函数。
ReadInput 从指定的输入文件中读取系统控制和模拟所需的参数，将其存储在 InputControl 和 InputPara 类的对象中；
PrintInput打印输入参数。
依赖：依赖于 Class.h 中定义的 InputControl 和 InputPara 类。
*/
#ifndef INPUT_H_
#define INPUT_H_

#include <iostream>
#include <fstream>
#include <string>
#include "Class.h"

void ReadInput(std::string InputFileName, InputControl& IC, InputPara& IP);

void PrintInput(InputControl IC, InputPara IP);

#endif
