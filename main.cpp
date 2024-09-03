#include <bits/stdc++.h>
using namespace std;
#define append push_back
int main(){
    int n; cin>>n;
    vector<int> v(n); 
    for(auto &val:v) cin >> val;
    int i = lower_bound(v.begin(), v.end(), 20)-v.begin();
    cout<<i<<endl;
    return 0;
}
