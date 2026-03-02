from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# Views

def base(request):
    symbols = request.session.get('symbols', [])
    return render(request, '1/base.html', {'symbols': symbols})

def bitcoin(request):
    symbols = request.session.get('symbols', [])
    return render(request, '2/fire.html', {'symbols': symbols})

def ethereum(request):
    symbols = request.session.get('symbols', [])
    return render(request, '2/smoke.html', {'symbols': symbols})
@csrf_exempt
def update_sidebar(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        symbol = data.get('symbol')
        symbols = request.session.get('symbols', [])
        if symbol not in symbols:
            symbols.append(symbol)
            request.session['symbols'] = symbols
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})
