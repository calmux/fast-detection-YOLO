
#pragma once
#ifdef YOLODLL_EXPORTS
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllexport) 
#else
#define YOLODLL_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllimport) 
#else
#define YOLODLL_API
#endif