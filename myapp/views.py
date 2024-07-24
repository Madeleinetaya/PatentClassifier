from django.shortcuts import render, redirect
from django.contrib.auth import login as auth_login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from .forms import SignUpForm, CustomAuthenticationForm, ClassificationForm
from .models import Classification, ClassificationDetail
from .scripts.mon_notebook import fonction_pour_les_mots_ayant_influer_le_plus_la_prediction

def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect('home')
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect('home')
    else:
        form = CustomAuthenticationForm()
    return render(request, 'login.html', {'form': form})

@login_required
def classify_list(request):
    classifications = Classification.objects.filter(user=request.user)
    return render(request, 'classify_list.html', {'classifications': classifications})

@login_required
def classify_create(request):
    if request.method == 'POST':
        form = ClassificationForm(request.POST, request.FILES)
        if form.is_valid():
            classification = form.save(commit=False)
            classification.user = request.user
            classification.save()
            result = fonction_pour_les_mots_ayant_influer_le_plus_la_prediction(classification.description, classification.depth)
            ##ClassificationDetail.objects.create(classification=classification, detail=result)
            return render(request, 'classify_results.html', {'results': result})
    else:
        form = ClassificationForm()
    return render(request, 'classify_form.html', {'form': form})

@login_required
def classify_detail(request, pk):
    classification = Classification.objects.get(pk=pk)
    return render(request, 'classify_detail.html', {'classification': classification})


def logout_view(request):
    logout(request)
    return redirect('home')

def home(request):
    return render(request, 'home.html')
