#include <iostream>
#include <math.h>
#include <vector>
#include <functional>
#include <memory>
#include <set>
#include <algorithm>
#include <typeinfo>

template <class T>
class Value
{
public:
    T data;
    float grad = 0;
    std::string _op;
    std::string label;
    std::vector<Value *> _prev;
    std::function<void()> _backward = []() {};

    Value(T data, std::string label = "_", std::vector<Value *> _children = {}, std::string _op = "_")
        : data(data), label(label), _prev(_children), _op(_op) {}

    Value operator+(Value &other)
    {
        Value out(this->data + other.data, this->label + "+" + other.label, {this, &other}, "+");
        out._backward = [&]()
        {
            this->grad += out.grad;
            other.grad += out.grad;
        };
        return out;
    }

    Value operator+(int other_int)
    {
        static Value other = Value(other_int, "const " + std::to_string(other_int));

        Value out(this->data + other.data, this->label + "+" + other.label, {this, &other}, "+");
        out._backward = [&]()
        {
            this->grad += out.grad;
            other.grad += out.grad;
        };
        return out;
    }

    Value operator+(float other_float)
    {
        static Value other = Value(other_float, "const " + std::to_string(other_float));

        Value out(this->data + other.data, this->label + "+" + other.label, {this, &other}, "+");
        out._backward = [&]()
        {
            this->grad += out.grad;
            other.grad += out.grad;
        };
        return out;
    }

    friend Value operator+(int other_int, Value &value)
    {
        return value + other_int;
    }

    friend Value operator+(float other_float, Value &value)
    {
        return value + other_float;
    }

    Value operator*(Value &other)
    {
        Value out(this->data * other.data, this->label + "*" + other.label, {this, &other}, "*");

        out._backward = [&]()
        {
            this->grad += other.data * out.grad;
            other.grad += this->data * out.grad;
        };

        return out;
    }

    Value operator*(int other_int)
    {
        static Value other = Value(other_int, "const " + std::to_string(other_int));

        Value out(this->data * other.data, this->label + "*" + other.label, {this, &other}, "*");

        out._backward = [&]()
        {
            this->grad += other.data * out.grad;
            other.grad += this->data * out.grad;
        };

        return out;
    }

    Value operator*(float other_float)
    {
        static Value other = Value(other_float, "const " + std::to_string(other_float));

        Value out(this->data * other.data, this->label + "*" + other.label, {this, &other}, "*");

        out._backward = [&]()
        {
            this->grad += other.data * out.grad;
            other.grad += this->data * out.grad;
        };

        return out;
    }

    friend Value operator*(int other_int, Value &value)
    {
        return value * other_int;
    }

    friend Value operator*(float other_float, Value &value)
    {
        return value * other_float;
    }

    Value operator/(Value &other)
    {
        static Value pow = other.pow_(-1);
        return *this * pow;
    }

    Value operator-()
    {
        this->data *= -1;
        this->label = "-" + this->label;
        return *this;
    }

    Value operator-(Value &other)
    {
        static Value neg_other = -other;
        return *this + neg_other;
    }

    Value pow_(float k)
    {
        Value out(pow(this->data, k), this->label + "^" + std::to_string(k), {this}, "pow");

        out._backward = [&]()
        { this->grad += k * pow(out.data, k - 1) * out.grad; };

        return out;
    }

    Value pow_(int k)
    {
        Value out(pow(this->data, k), "pow(" + this->label + "," + std::to_string(k) + ")", {this}, "pow");

        out._backward = [&]()
        { this->grad += k * pow(out.data, k - 1) * out.grad; };

        return out;
    }

    Value tanh()
    {
        float x = this->data, t;
        t = (exp(2 * x) - 1) / (exp(2 * x) + 1);
        Value out(t, "tanh(" + this->label + ")", {this}, "tanh");

        out._backward = [&]()
        { this->grad += (1 - t * t) * out.grad; };

        return out;
    }

    Value exp_()
    {
        Value out(exp(this->data), "exp(" + this->label + ")", {this}, "exp");

        out._backward = [&]()
        { this->grad += out.data * out.grad; };

        return out;
    }

    void backward()
    {
        std::vector<Value *> topo;
        std::set<Value *> visited;
        std::function<void(Value &)> b_topo = [&](Value &value)
        {
            const bool is_in = visited.find(&value) != visited.end();
            if (!is_in)
            {
                visited.insert(&value);
                for (auto child : value._prev)
                {
                    b_topo(*child);
                }
                topo.push_back(&value);
            }
        };
        b_topo(*this);

        std::reverse(topo.begin(), topo.end());
        this->grad = 1;
        for (int i = 0; i < topo.size(); ++i)
        {
            topo[i]->_backward();
        }
    }

    std::string str()
    {
        return this->label + "(data=" + std::to_string(this->data) + ", grad=" + std::to_string(this->grad) + ")";
    }

    void print()
    {
        std::vector<Value *> topo;
        std::set<Value *> visited;
        std::function<void(Value &)> b_topo = [&](Value &value)
        {
            const bool is_in = visited.find(&value) != visited.end();
            if (!is_in)
            {
                visited.insert(&value);
                for (auto child : value._prev)
                {
                    b_topo(*child);
                }
                topo.push_back(&value);
            }
        };
        b_topo(*this);

        std::reverse(topo.begin(), topo.end());
        this->grad = 1;
        for (int i = 0; i < topo.size(); ++i)
        {
            std::cout << topo[i]->str() << std::endl;
        }
    }
};

int main()
{
    // Value<float> x1(2, "x1"), x2(0, "x2"), w1(-3, "w1"), w2(1, "w2"), b(6.8813, "b");

    // Value<float> x1w1 = x1 * w1, x2w2 = x2 * w2;
    // Value<float> x1w1x2w2 = x1w1 + x2w2;
    // Value<float> n = x1w1x2w2 + b;
    // n.label = "n";
    // Value<float> out = n.tanh();
    // out.label = "out";
    // out.backward();
    // out.print();

    Value<float> a(2, "a"), b(4, "b");

    Value<float> c = a - b;
    // c.label = "c";
    // c.backward();
    c.print();

    return 0;
}