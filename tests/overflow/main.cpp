#include <cstring>
#include <iostream>

int main() {
    char buffer[8];
    strcpy(buffer, "0123456789ABCDEF");
    std::cout << buffer;
    return 0;
}