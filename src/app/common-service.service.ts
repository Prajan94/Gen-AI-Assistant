import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class CommonServiceService {

  constructor(private http: HttpClient) { }
  askAI(query: any, course: any): Observable<any>  {
    const params = new HttpParams()
    .set('query', query)
    .set('course', course)
    return this.http.get('http://127.0.0.1:5000/api/data', {params: params});
  }

  askEmailAI(query: any): Observable<any>  {
    const params = new HttpParams()
    .set('query', query)
    return this.http.get('http://127.0.0.1:5000/api/gmail', {params: params});
  }

    askFigmaAI(query: any): Observable<string> {
    const params = new HttpParams()
    .set('query', query)
    return this.http.get<string>('http://127.0.0.1:5000/api/figma', {params: params});
  }

  genSynData(query: any, rows: any): Observable<any> {
    const params = new HttpParams()
    .set('query', query)
    .set('count', parseInt(rows))
    return this.http.get<string>('http://127.0.0.1:5000/api/syngen', {params: params});
  }

  voiceAssistant(blob: any): Observable<any>  {
    const formdata = new FormData()
    formdata.append("audio", blob)
    // const params = new HttpParams()
    // .set('query', "voice")
    return this.http.post('http://127.0.0.1:5000/api/voice', formdata);

  }

}