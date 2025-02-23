#include "debug.h"

struct __FILE
{
	int handle;
};

FILE __stdout;


//定义_sys_exit函数避免使用半主机模式
void _sys_exit(int x)
{
	x = x;//这里的赋值没有实际意义，避免空函数
}

//重定义fputc
int fputc(int ch, FILE *f)
{ 
	while((USART2->SR&0X40)==0);//循环发送,直到发送完毕   
	USART2->DR=(uint8_t)ch;   
	return ch;
}


